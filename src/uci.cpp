#include "datagen.h"
#include "search.h"
#include "thread.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <thread>
#include <vector>

#define ENGINE_NAME "Halcyon"
#define ENGINE_VERSION "1.0"

namespace {

struct Engine {
    TT tt;
    std::vector<std::unique_ptr<Searcher>> searchers;
    std::atomic<bool> stop{false};
    BigThread worker;
    Position pos;
    std::vector<Key> history;
    int moveOverhead = 10;

    Engine() {
        tt.resize(64);
        setThreads(1);
        pos.setFen(STARTPOS);
        history.assign(1, pos.key);
    }
    ~Engine() { halt(); }

    void setThreads(int n) {
        halt();
        searchers.clear();
        for (int i = 0; i < n; i++) {
            searchers.push_back(std::make_unique<Searcher>());
            searchers.back()->clearHistory();
        }
    }
    void halt() {
        stop = true;
        if (worker.joinable()) worker.join();
    }
    void wait() {
        if (worker.joinable()) worker.join();
    }
    void newGame() {
        halt();
        tt.clear();
        for (auto& s : searchers) s->clearHistory();
    }
    void go(const Limits& lim, bool silent = false) {
        halt();
        stop = false;
        tt.newSearch();
        for (size_t i = 0; i < searchers.size(); i++) {
            Searcher& s = *searchers[i];
            s.tt = &tt;
            s.stop = &stop;
            s.isMain = i == 0;
            s.pool = &searchers;
            s.silent = silent || i != 0;
            s.lim = lim;
            s.setRoot(pos, history);
        }
        worker.start([this, silent] {
            std::vector<BigThread> helpers;
            for (size_t i = 1; i < searchers.size(); i++) helpers.emplace_back([this, i] { searchers[i]->think(); });
            searchers[0]->think();
            stop = true;
            for (auto& h : helpers) h.join();
            if (!silent) {
                printf("bestmove %s\n", moveToStr(searchers[0]->bestMove).c_str());
                fflush(stdout);
            }
        });
    }
    u64 totalNodes() const {
        u64 n = 0;
        for (auto& s : searchers) n += s->nodeCount();
        return n;
    }
};

const char* BenchFens[] = {
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/R3KB1R w KQ - 3 9",
    "2rq1rk1/pb1n1ppN/4p3/1pb5/3P1Pn1/P1N5/1PQ1B1PP/R1B2RK1 b - - 0 16",
    "r2q1rk1/1b1nbppp/p2ppn2/1p6/3NPP2/1BN1B3/PPP1Q1PP/R4RK1 w - - 2 12",
    "8/5pk1/6p1/3R3p/7P/5PP1/1r6/6K1 b - - 4 40",
    "6k1/5pp1/7p/3n4/8/1B4P1/5P1P/6K1 w - - 0 35",
    "r1b2rk1/2q1bppp/p2p1n2/np2p3/3PP3/2P2N1P/PPBN1PP1/R1BQR1K1 b - - 1 13",
    "3r1rk1/1pq1bppp/p1n1pn2/8/P2P4/1BN1BN2/1P2QPPP/3R1RK1 w - - 3 17",
    "8/8/4kpp1/3p1b2/p6P/2B5/6P1/6K1 w - - 0 50",
    "4r1k1/1p3ppp/p1p5/8/2P1n3/1P2R1P1/P4P1P/2B3K1 w - - 2 26",
    "r3r1k1/pp3pbp/1qp3p1/2B5/2BP2b1/Q1n2N2/P4PPP/3R1K1R w - - 2 18",
    "8/1p4k1/p2p2p1/P1pPb2p/2P1P2P/1P4P1/5K2/3B4 w - - 0 45",
    "2r3k1/5pp1/p2p3p/1p1Pp2P/5q2/1P3P2/PKP1Q3/3R4 b - - 3 30",
};

void bench(Engine& eng, int depth) {
    u64 nodes = 0;
    i64 t0 = nowMs();
    for (const char* fen : BenchFens) {
        eng.newGame();
        eng.pos.setFen(fen);
        eng.history.assign(1, eng.pos.key);
        Limits lim;
        lim.start = nowMs();
        lim.depth = depth;
        eng.go(lim, true);
        eng.wait();
        nodes += eng.totalNodes();
    }
    i64 ms = std::max<i64>(nowMs() - t0, 1);
    printf("%llu nodes %llu nps\n", (unsigned long long)nodes, (unsigned long long)(nodes * 1000 / ms));
    fflush(stdout);
}

void setPosition(Engine& eng, std::istringstream& is) {
    std::string tok, fen;
    is >> tok;
    if (tok == "startpos") {
        fen = STARTPOS;
        is >> tok; // "moves", if present
    } else if (tok == "fen") {
        while (is >> tok && tok != "moves") fen += tok + " ";
    } else
        return;
    eng.pos.setFen(fen);
    eng.history.assign(1, eng.pos.key);
    while (is >> tok) {
        Move m = parseMove(eng.pos, tok);
        if (!m) break;
        eng.pos.makeMove(m);
        if (eng.pos.rule50 == 0) eng.history.clear();
        eng.history.push_back(eng.pos.key);
    }
}

void go(Engine& eng, std::istringstream& is) {
    Limits lim;
    lim.start = nowMs();
    i64 time[2] = {-1, -1}, inc[2] = {0, 0}, movetime = -1;
    int mtg = 0;
    std::string tok;
    while (is >> tok) {
        if (tok == "wtime") is >> time[WHITE];
        else if (tok == "btime") is >> time[BLACK];
        else if (tok == "winc") is >> inc[WHITE];
        else if (tok == "binc") is >> inc[BLACK];
        else if (tok == "movestogo") is >> mtg;
        else if (tok == "movetime") is >> movetime;
        else if (tok == "depth") is >> lim.depth;
        else if (tok == "nodes") is >> lim.hardNodes;
    }
    lim.depth = std::clamp(lim.depth, 1, MAX_PLY - 1);
    const int us = eng.pos.stm;
    if (movetime >= 0)
        lim.softMs = lim.hardMs = std::max<i64>(1, movetime - eng.moveOverhead);
    else if (time[us] >= 0) {
        const double t = std::max<double>(1, time[us] - eng.moveOverhead);
        double soft = mtg > 0 ? 0.9 * t / (mtg + 1) + 0.5 * inc[us] : t / 22 + 0.5 * inc[us];
        double hard = std::min(t * 0.5, soft * 5);
        lim.softMs = (i64)std::min(soft, hard);
        lim.hardMs = (i64)hard;
    }
    eng.go(lim);
}

} // namespace

void uciLoop(int argc, char** argv) {
    Engine eng;
    auto handle = [&](const std::string& line) -> bool {
        std::istringstream is(line);
        std::string cmd;
        is >> cmd;
        if (cmd == "uci") {
            printf("id name " ENGINE_NAME " " ENGINE_VERSION "\nid author Claude\n");
            printf("option name Hash type spin default 64 min 1 max 65536\n");
            printf("option name Threads type spin default 1 min 1 max 256\n");
            printf("option name MoveOverhead type spin default 10 min 0 max 5000\n");
            printf("option name EvalFile type string default <embedded>\n");
            printf("uciok\n");
        } else if (cmd == "isready")
            printf("readyok\n");
        else if (cmd == "ucinewgame")
            eng.newGame();
        else if (cmd == "position") {
            eng.halt();
            setPosition(eng, is);
        } else if (cmd == "go")
            go(eng, is);
        else if (cmd == "stop")
            eng.halt();
        else if (cmd == "quit")
            return false;
        else if (cmd == "setoption") {
            std::string tok, name, value;
            is >> tok >> name >> tok;
            std::getline(is >> std::ws, value);
            std::transform(name.begin(), name.end(), name.begin(), ::tolower);
            if (name == "hash") {
                eng.halt();
                eng.tt.resize(std::max(1, atoi(value.c_str())));
            } else if (name == "threads")
                eng.setThreads(std::clamp(atoi(value.c_str()), 1, 256));
            else if (name == "moveoverhead")
                eng.moveOverhead = atoi(value.c_str());
            else if (name == "evalfile") {
                eng.halt();
                if (!NNUE::loadFile(value.c_str())) printf("info string failed to load network %s\n", value.c_str());
            }
        } else if (cmd == "bench") {
            int depth = 13;
            is >> depth;
            bench(eng, depth);
        } else if (cmd == "perft") {
            int depth = 5;
            is >> depth;
            i64 t0 = nowMs();
            u64 n = perft(eng.pos, depth);
            i64 ms = std::max<i64>(nowMs() - t0, 1);
            printf("%llu nodes %llu nps\n", (unsigned long long)n, (unsigned long long)(n * 1000 / ms));
        } else if (cmd == "eval") {
            eng.halt();
            Searcher& s = *eng.searchers[0];
            if (NNUE::loaded) s.nn.reset();
            s.stack[6].ply = 0;
            printf("eval %d (%s)\n", eng.pos.checkers ? 0 : s.evaluate(eng.pos, &s.stack[6]), NNUE::loaded ? "nnue" : "hce");
        } else if (cmd == "checknet") {
            // checknet <datafile> <count>: print the network output for the first records of a data file
            std::string path;
            int count = 8;
            is >> path >> count;
            FILE* f = fopen(path.c_str(), "rb");
            PackedPos p;
            Searcher& s = *eng.searchers[0];
            for (int i = 0; f && i < count && fread(&p, sizeof(p), 1, f) == 1; i++) {
                std::string rows[8];
                int n = 0, sq = 0;
                char grid[64];
                memset(grid, 0, sizeof(grid));
                for (Bitboard b = p.occ; b; n++) {
                    sq = poplsb(b);
                    grid[sq] = "PNBRQKpnbrqk"[(p.pcs[n / 2] >> (4 * (n & 1))) & 15];
                }
                std::string fen;
                for (int r = 7; r >= 0; r--) {
                    int empty = 0;
                    for (int fl = 0; fl < 8; fl++) {
                        char c = grid[r * 8 + fl];
                        if (!c) empty++;
                        else {
                            if (empty) fen += char('0' + empty);
                            empty = 0;
                            fen += c;
                        }
                    }
                    if (empty) fen += char('0' + empty);
                    if (r) fen += '/';
                }
                fen += p.stm ? " b - - 0 1" : " w - - 0 1";
                Position pos;
                pos.setFen(fen);
                s.nn.reset();
                s.stack[6].ply = 0;
                printf("check %d: net %d cp (data score %d)  %s\n", i, NNUE::loaded ? s.nn.evaluate(pos, 0) : 0, p.stm ? -p.score : p.score, fen.c_str());
            }
            if (f) fclose(f);
        } else if (cmd == "d")
            printf("%s\nkey %016llx\n", eng.pos.fen().c_str(), (unsigned long long)eng.pos.key);
        else if (cmd == "datagen") {
            // datagen <threads> <positions> <nodes> <outfile> [seed]
            int threads = 1;
            u64 positions = 1000000, nodes = 5000, seed = (u64)nowMs();
            std::string out = "data.bin";
            is >> threads >> positions >> nodes >> out >> seed;
            runDatagen(threads, positions, nodes, out, seed);
        } else if (cmd == "genfens") {
            // genfens <count> <seed> <outfile>
            int count = 1000;
            u64 seed = 1;
            std::string out = "book.epd";
            is >> count >> seed >> out;
            runGenFens(count, seed, out);
        }
        fflush(stdout);
        return true;
    };

    if (argc > 1) { // run command-line arguments as a single command, then exit
        std::string line;
        for (int i = 1; i < argc; i++) line += std::string(argv[i]) + " ";
        handle(line);
        eng.wait();
        return;
    }
    std::string line;
    while (std::getline(std::cin, line))
        if (!handle(line)) break;
    eng.halt();
}
