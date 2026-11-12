// Engine-vs-engine match runner with SPRT, used to accept or reject every engine change.
//
//   match -e1 ./new -e2 ./old -book book.epd -tc 4+0.04 -games 2000 -c 8 [-sprt 0 5]
//         [-o1 "EvalFile=a.bin;Hash=16"] [-o2 ...] [-hash 16]
//
// Each opening is played twice with colours reversed. Results are reported from e1's point of view.
#include "../src/position.h"
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <poll.h>
#include <signal.h>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>
#include <algorithm>
#include <sys/wait.h>

using Clock = std::chrono::steady_clock;

struct EngineProc {
    pid_t pid = -1;
    int in = -1, out = -1;
    std::string buf;

    bool start(const std::string& path) {
        int a[2], b[2];
        if (pipe(a) || pipe(b)) return false;
        pid = fork();
        if (pid == 0) {
            dup2(a[0], 0);
            dup2(b[1], 1);
            close(a[0]), close(a[1]), close(b[0]), close(b[1]);
            execl(path.c_str(), path.c_str(), (char*)nullptr);
            _exit(127);
        }
        close(a[0]);
        close(b[1]);
        in = a[1];
        out = b[0];
        return pid > 0;
    }
    void send(const std::string& s) {
        std::string l = s + "\n";
        if (write(in, l.data(), l.size()) < 0) {}
    }
    // Reads one line; false on timeout or a dead engine.
    bool readLine(std::string& line, int timeoutMs) {
        auto deadline = Clock::now() + std::chrono::milliseconds(timeoutMs);
        for (;;) {
            size_t nl = buf.find('\n');
            if (nl != std::string::npos) {
                line = buf.substr(0, nl);
                buf.erase(0, nl + 1);
                return true;
            }
            int left = (int)std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count();
            if (left <= 0) return false;
            pollfd p{out, POLLIN, 0};
            if (poll(&p, 1, left) <= 0) return false;
            char tmp[4096];
            ssize_t n = read(out, tmp, sizeof(tmp));
            if (n <= 0) return false;
            buf.append(tmp, n);
        }
    }
    bool waitFor(const std::string& token, int timeoutMs) {
        std::string l;
        while (readLine(l, timeoutMs))
            if (l.rfind(token, 0) == 0) return true;
        return false;
    }
    void quit() {
        if (pid > 0) {
            send("quit");
            close(in);
            close(out);
            kill(pid, SIGKILL);
            waitpid(pid, nullptr, 0);
            pid = -1;
        }
    }
};

struct Config {
    std::string e[2], opts[2], book;
    double base = 4, inc = 0.04;
    int games = 1000, concurrency = 8, hash = 16;
    bool sprt = false;
    double elo0 = 0, elo1 = 5;
} cfg;

static std::vector<std::string> openings;
static std::mutex mtx;
static std::atomic<int> nextPair{0};
static std::atomic<bool> done{false};
static int W = 0, L = 0, D = 0, penta[5], timeLosses = 0, crashes = 0;

static bool setup(EngineProc& p, int idx) {
    if (!p.start(cfg.e[idx])) return false;
    p.send("uci");
    if (!p.waitFor("uciok", 10000)) return false;
    p.send("setoption name Hash value " + std::to_string(cfg.hash));
    std::istringstream ss(cfg.opts[idx]);
    std::string kv;
    while (std::getline(ss, kv, ';')) {
        size_t eq = kv.find('=');
        if (eq != std::string::npos) p.send("setoption name " + kv.substr(0, eq) + " value " + kv.substr(eq + 1));
    }
    p.send("isready");
    return p.waitFor("readyok", 30000);
}

// Returns the result for white: 2 win, 1 draw, 0 loss; -1 on engine failure.
static int playGame(EngineProc* eng[2], const std::string& fen, int& loserFault) {
    Position pos;
    pos.setFen(fen);
    std::vector<Key> hist{pos.key};
    std::string moves;
    double clock[2] = {cfg.base * 1000, cfg.base * 1000};
    int winPlies = 0, drawPlies = 0, winSign = 0;
    loserFault = 0;
    for (int i = 0; i < 2; i++) {
        eng[i]->send("ucinewgame");
        eng[i]->send("isready");
        if (!eng[i]->waitFor("readyok", 30000)) return -1;
    }
    for (int ply = 0;; ply++) {
        MoveList ml;
        generateLegal(pos, ml);
        if (!ml.n) return pos.checkers ? (pos.stm == WHITE ? 0 : 2) : 1;
        int rep = 0;
        for (Key k : hist) rep += k == pos.key;
        if (pos.rule50 >= 100 || pos.insufficientMaterial() || rep >= 3) return 1;

        EngineProc& e = *eng[pos.stm];
        e.send("position fen " + fen + (moves.empty() ? "" : " moves" + moves));
        char go[160];
        snprintf(go, sizeof(go), "go wtime %d btime %d winc %d binc %d", (int)clock[0], (int)clock[1], (int)(cfg.inc * 1000), (int)(cfg.inc * 1000));
        auto t0 = Clock::now();
        e.send(go);
        std::string line, best;
        int score = 0;
        bool haveScore = false;
        for (;;) {
            if (!e.readLine(line, (int)clock[pos.stm] + 2000)) {
                loserFault = 2;
                return -1;
            }
            if (line.rfind("bestmove", 0) == 0) {
                std::istringstream is(line);
                is >> best >> best;
                break;
            }
            size_t sp = line.find(" score ");
            if (sp != std::string::npos) {
                std::istringstream is(line.substr(sp + 7));
                std::string kind;
                int v;
                is >> kind >> v;
                score = kind == "mate" ? (v > 0 ? 30000 : -30000) : v;
                haveScore = true;
            }
        }
        double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        clock[pos.stm] -= ms;
        if (clock[pos.stm] < 0) {
            loserFault = 1;
            return pos.stm == WHITE ? 0 : 2;
        }
        clock[pos.stm] += cfg.inc * 1000;
        Move m = parseMove(pos, best);
        if (!m) {
            loserFault = 2;
            fprintf(stderr, "illegal move %s in %s\n", best.c_str(), pos.fen().c_str());
            return pos.stm == WHITE ? 0 : 2;
        }
        if (haveScore) {
            int white = pos.stm == WHITE ? score : -score;
            int sign = white >= 1000 ? 1 : white <= -1000 ? -1 : 0;
            winPlies = sign && sign == winSign ? winPlies + 1 : (sign ? 1 : 0);
            winSign = sign;
            drawPlies = std::abs(score) <= 8 && ply >= 70 ? drawPlies + 1 : 0;
            if (winPlies >= 6) return winSign > 0 ? 2 : 0;
            if (drawPlies >= 12) return 1;
        }
        pos.makeMove(m);
        if (pos.rule50 == 0) hist.clear();
        hist.push_back(pos.key);
        moves += " " + best;
    }
}

static double eloOf(double s) { return -400.0 * std::log10(1.0 / std::min(std::max(s, 1e-6), 1 - 1e-6) - 1.0); }

static void report(bool final) {
    int n = W + L + D, pairs = 0;
    for (int p : penta) pairs += p;
    if (!n || !pairs) return;
    double mu = 0, var = 0;
    for (int i = 0; i < 5; i++) mu += penta[i] * (i / 4.0);
    mu /= pairs;
    for (int i = 0; i < 5; i++) var += penta[i] * (i / 4.0 - mu) * (i / 4.0 - mu);
    var = std::max(var / pairs, 1e-9);
    double se = std::sqrt(var / pairs);
    double elo = eloOf(mu), lo = eloOf(mu - 1.96 * se), hi = eloOf(mu + 1.96 * se);
    printf("%sgames %d  +%d -%d =%d  score %.1f%%  elo %+.1f +/- %.1f  penta [%d %d %d %d %d]", final ? "FINAL " : "", n, W, L, D, 100.0 * (W + D / 2.0) / n, elo,
           (hi - lo) / 2, penta[0], penta[1], penta[2], penta[3], penta[4]);
    if (timeLosses || crashes) printf("  timeouts %d crashes %d", timeLosses, crashes);
    if (cfg.sprt) {
        double s0 = 1 / (1 + std::pow(10, -cfg.elo0 / 400)), s1 = 1 / (1 + std::pow(10, -cfg.elo1 / 400));
        double llr = pairs * (s1 - s0) * (2 * mu - s0 - s1) / (2 * var);
        const double bound = std::log(0.95 / 0.05);
        printf("  LLR %.2f (%.2f, %.2f)", llr, -bound, bound);
        if (llr >= bound) printf("  H1 accepted"), done = true;
        if (llr <= -bound) printf("  H0 accepted"), done = true;
    }
    printf("\n");
    fflush(stdout);
}

static void worker() {
    EngineProc p[2];
    if (!setup(p[0], 0) || !setup(p[1], 1)) {
        fprintf(stderr, "failed to start engines\n");
        done = true;
        return;
    }
    while (!done) {
        int pair = nextPair++;
        if (pair * 2 >= cfg.games) break;
        const std::string& fen = openings[pair % openings.size()];
        double pairScore = 0;
        bool valid = true;
        int res[2];
        for (int g = 0; g < 2 && valid; g++) {
            EngineProc* eng[2] = {&p[g], &p[g ^ 1]}; // game 0: e1 is white
            int fault = 0;
            int r = playGame(eng, fen, fault);
            if (r < 0) { // engine died or hung: restart both and drop the pair
                std::lock_guard<std::mutex> lock(mtx);
                crashes++;
                valid = false;
                p[0].quit(), p[1].quit();
                if (!setup(p[0], 0) || !setup(p[1], 1)) done = true;
                break;
            }
            if (fault == 1) {
                std::lock_guard<std::mutex> lock(mtx);
                timeLosses++;
            }
            res[g] = g == 0 ? r : 2 - r; // from e1's point of view
            pairScore += res[g] / 2.0;
        }
        if (!valid) continue;
        std::lock_guard<std::mutex> lock(mtx);
        for (int g = 0; g < 2; g++) (res[g] == 2 ? W : res[g] == 1 ? D : L)++;
        penta[(int)std::lround(pairScore * 2)]++;
        if ((W + L + D) % 100 == 0 || cfg.sprt) {
            static int lastPrinted = 0;
            if (W + L + D - lastPrinted >= 100 || done) lastPrinted = W + L + D, report(false);
            else if (cfg.sprt) {
                // evaluate the stopping rule silently
                int pairs = 0;
                for (int x : penta) pairs += x;
                double mu = 0, var = 0;
                for (int i = 0; i < 5; i++) mu += penta[i] * (i / 4.0);
                mu /= pairs;
                for (int i = 0; i < 5; i++) var += penta[i] * (i / 4.0 - mu) * (i / 4.0 - mu);
                var = std::max(var / pairs, 1e-9);
                double s0 = 1 / (1 + std::pow(10, -cfg.elo0 / 400)), s1 = 1 / (1 + std::pow(10, -cfg.elo1 / 400));
                double llr = pairs * (s1 - s0) * (2 * mu - s0 - s1) / (2 * var);
                if (pairs >= 20 && std::fabs(llr) >= std::log(0.95 / 0.05)) done = true;
            }
        }
    }
    p[0].quit();
    p[1].quit();
}

int main(int argc, char** argv) {
    signal(SIGPIPE, SIG_IGN);
    initBitboards();
    initZobrist();
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto val = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
        if (a == "-e1") cfg.e[0] = val();
        else if (a == "-e2") cfg.e[1] = val();
        else if (a == "-o1") cfg.opts[0] = val();
        else if (a == "-o2") cfg.opts[1] = val();
        else if (a == "-book") cfg.book = val();
        else if (a == "-games") cfg.games = atoi(val().c_str());
        else if (a == "-c") cfg.concurrency = atoi(val().c_str());
        else if (a == "-hash") cfg.hash = atoi(val().c_str());
        else if (a == "-tc") {
            std::string t = val();
            size_t plus = t.find('+');
            cfg.base = atof(t.substr(0, plus).c_str());
            cfg.inc = plus == std::string::npos ? 0 : atof(t.substr(plus + 1).c_str());
        } else if (a == "-sprt") {
            cfg.sprt = true;
            cfg.elo0 = atof(val().c_str());
            cfg.elo1 = atof(val().c_str());
        }
    }
    std::ifstream bf(cfg.book);
    for (std::string l; std::getline(bf, l);)
        if (!l.empty()) openings.push_back(l);
    if (cfg.e[0].empty() || cfg.e[1].empty() || openings.empty()) {
        fprintf(stderr, "usage: match -e1 A -e2 B -book book.epd [-tc 4+0.04] [-games N] [-c N] [-sprt e0 e1] [-o1 opts] [-o2 opts]\n");
        return 1;
    }
    // Start at a random offset in the book so separate runs see different openings.
    nextPair = 0;
    srand((unsigned)time(nullptr));
    std::rotate(openings.begin(), openings.begin() + rand() % openings.size(), openings.end());
    std::vector<std::thread> pool;
    for (int i = 0; i < cfg.concurrency; i++) pool.emplace_back(worker);
    for (auto& t : pool) t.join();
    report(true);
    return 0;
}
