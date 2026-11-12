// Self-play training data generation and opening generation.
#include "datagen.h"
#include "search.h"
#include "thread.h"
#include <algorithm>
#include <cstdio>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace {

struct Rng {
    u64 s;
    u64 next() {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        return s * 0x2545F4914F6CDD1DULL;
    }
};

struct Worker {
    TT tt;
    std::atomic<bool> stop{false};
    std::unique_ptr<Searcher> s = std::make_unique<Searcher>();
    Rng rng;
    Position pos;
    std::vector<Key> hist;

    Worker(u64 seed, int ttMb) : rng{seed * 0x9E3779B97F4A7C15ULL + 0x1234567ULL} {
        for (int i = 0; i < 16; i++) rng.next();
        tt.resize(ttMb);
        s->tt = &tt;
        s->stop = &stop;
        s->silent = true;
        s->clearHistory();
    }

    void search(u64 softNodes, u64 hardNodes) {
        stop = false;
        tt.newSearch();
        s->lim = Limits{};
        s->lim.start = nowMs();
        s->lim.softNodes = softNodes;
        s->lim.hardNodes = hardNodes;
        s->setRoot(pos, hist);
        s->think();
    }

    // Plays random plies from the start position; false if the game ended on the way.
    bool randomOpening(int plies) {
        pos.setFen(STARTPOS);
        hist.assign(1, pos.key);
        for (int i = 0; i < plies; i++) {
            MoveList ml;
            generateLegal(pos, ml);
            if (!ml.n) return false;
            pos.makeMove(ml.moves[rng.next() % ml.n]);
            hist.push_back(pos.key);
        }
        MoveList ml;
        generateLegal(pos, ml);
        return ml.n > 0;
    }

    bool threefold() const {
        int n = 0;
        for (Key k : hist) n += k == pos.key;
        return n >= 3;
    }
};

} // namespace

void runDatagen(int threads, u64 target, u64 nodes, const std::string& out, u64 seed) {
    FILE* f = fopen(out.c_str(), "ab");
    if (!f) {
        printf("cannot open %s\n", out.c_str());
        return;
    }
    std::mutex mtx;
    std::atomic<u64> total{0}, games{0};
    u64 results[3] = {0, 0, 0};
    const i64 t0 = nowMs();
    auto work = [&](int id) {
        Worker w(seed * 1000 + id, 16);
        std::vector<PackedPos> rec;
        while (total.load() < target) {
            if (!w.randomOpening(8 + int(w.rng.next() & 1))) continue;
            w.tt.clear();
            w.s->clearHistory();
            w.search(nodes, nodes * 8);
            if (std::abs(w.s->bestScore) > 300) continue;
            rec.clear();
            int result = -1, winCnt = 0, drawCnt = 0;
            for (int ply = 0; result < 0; ply++) {
                MoveList ml;
                generateLegal(w.pos, ml);
                if (!ml.n) {
                    result = w.pos.checkers ? (w.pos.stm == WHITE ? 0 : 2) : 1;
                    break;
                }
                if (w.pos.rule50 >= 100 || w.pos.insufficientMaterial() || w.threefold() || ply >= 400) {
                    result = 1;
                    break;
                }
                w.search(nodes, nodes * 8);
                const Move best = w.s->bestMove;
                const int score = w.s->bestScore, white = w.pos.stm == WHITE ? score : -score;
                if (std::abs(score) >= MATE_BOUND) {
                    result = white > 0 ? 2 : 0;
                    break;
                }
                winCnt = std::abs(score) >= 2000 ? winCnt + 1 : 0;
                drawCnt = std::abs(score) <= 6 && ply >= 60 ? drawCnt + 1 : 0;
                if (winCnt >= 5) result = white > 0 ? 2 : 0;
                if (drawCnt >= 12) result = 1;
                if (!w.pos.checkers && !isNoisy(best)) rec.push_back(packPosition(w.pos, white));
                w.pos.makeMove(best);
                w.hist.push_back(w.pos.key);
            }
            for (auto& r : rec) r.result = u8(result);
            std::lock_guard<std::mutex> lock(mtx);
            fwrite(rec.data(), sizeof(PackedPos), rec.size(), f);
            results[result]++;
            u64 before = total.fetch_add(rec.size());
            u64 g = ++games;
            if ((before + rec.size()) / 500000 != before / 500000) {
                fflush(f);
                double secs = (nowMs() - t0) / 1000.0;
                printf("%llu positions, %llu games (W %.1f%% D %.1f%% L %.1f%%), %.0f pos/s\n", (unsigned long long)(before + rec.size()), (unsigned long long)g,
                       100.0 * results[2] / g, 100.0 * results[1] / g, 100.0 * results[0] / g, (before + rec.size()) / secs);
                fflush(stdout);
            }
        }
    };
    std::vector<BigThread> pool;
    for (int i = 0; i < threads; i++) pool.emplace_back([&work, i] { work(i); });
    for (auto& t : pool) t.join();
    fclose(f);
    printf("done: %llu positions in %llu games\n", (unsigned long long)total.load(), (unsigned long long)games.load());
}

// Writes balanced random openings (one FEN per line) for engine-vs-engine testing.
void runGenFens(int count, u64 seed, const std::string& out) {
    FILE* f = fopen(out.c_str(), "w");
    if (!f) return;
    Worker w(seed, 16);
    for (int n = 0; n < count;) {
        if (!w.randomOpening(8 + int(w.rng.next() & 1))) continue;
        w.tt.clear();
        w.s->clearHistory();
        w.search(30000, 240000);
        if (std::abs(w.s->bestScore) > 60) continue;
        fprintf(f, "%s\n", w.pos.fen().c_str());
        n++;
    }
    fclose(f);
    printf("wrote %d openings to %s\n", count, out.c_str());
}
