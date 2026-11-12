// Move generator validation: perft node counts against known values, plus a
// consistency check that isPseudoLegal() agrees with the generator for every
// possible 16-bit move encoding.
#include "../src/position.h"
#include <chrono>
#include <cstdio>

static u64 checked = 0;
static bool fuzz(const Position& pos, int depth) {
    MoveList ml;
    generate(pos, ml, GEN_ALL);
    static thread_local bool in[65536];
    for (int i = 0; i < ml.n; i++) in[ml.moves[i]] = true;
    bool ok = true;
    for (int m = 0; m < 65536; m++) {
        if (pos.isPseudoLegal(Move(m)) != in[m]) {
            printf("isPseudoLegal mismatch: %s move %s flag %d (generated=%d)\n", pos.fen().c_str(), moveToStr(Move(m)).c_str(), flagOf(Move(m)), in[m]);
            ok = false;
        }
    }
    checked++;
    for (int i = 0; i < ml.n; i++) in[ml.moves[i]] = false;
    // noisy + quiet generation must equal full generation
    MoveList a, b;
    generate(pos, a, GEN_NOISY);
    generate(pos, b, GEN_QUIET);
    if (a.n + b.n != ml.n) {
        printf("split generation mismatch: %s\n", pos.fen().c_str());
        ok = false;
    }
    for (int i = 0; i < a.n; i++)
        if (!isNoisy(a.moves[i])) ok = false;
    for (int i = 0; i < b.n; i++)
        if (isNoisy(b.moves[i])) ok = false;
    if (depth > 0)
        for (int i = 0; i < ml.n && ok; i++) {
            if (!pos.isLegal(ml.moves[i])) continue;
            Position next = pos;
            next.makeMove(ml.moves[i]);
            Position fresh;
            fresh.setFen(next.fen());
            if (fresh.key != next.key || fresh.pawnKey != next.pawnKey || fresh.checkers != next.checkers || fresh.pinned != next.pinned) {
                printf("incremental state mismatch after %s in %s\n", moveToStr(ml.moves[i]).c_str(), pos.fen().c_str());
                ok = false;
            }
            ok = ok && fuzz(next, depth - 1);
        }
    return ok;
}

int main() {
    initBitboards();
    initZobrist();
    struct T {
        const char* fen;
        int depth;
        u64 nodes;
    } tests[] = {
        {STARTPOS, 6, 119060324ULL},
        {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 5, 193690690ULL},
        {"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 6, 11030083ULL},
        {"r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 5, 15833292ULL},
        {"r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1", 5, 15833292ULL},
        {"rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 5, 89941194ULL},
        {"r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 5, 164075551ULL},
        {"8/8/1k6/2b5/2pP4/8/5K2/8 b - d3 0 1", 6, 1440467ULL},
        {"8/8/8/8/k1p4R/8/3P4/3K4 w - - 0 1", 6, 1134888ULL},
        {"3k4/3p4/8/K1P4r/8/8/8/8 b - - 0 1", 6, 1134888ULL},
        {"r3k2r/1b4bq/8/8/8/8/7B/R3K2R w KQkq - 0 1", 4, 1274206ULL},
        {"2K2r2/4P3/8/8/8/8/8/3k4 w - - 0 1", 6, 3821001ULL},
        {"4k3/1P6/8/8/8/8/K7/8 w - - 0 1", 6, 217342ULL},
        {"8/8/2k5/5q2/5n2/8/5K2/8 b - - 0 1", 4, 23527ULL},
    };
    bool allOk = true;
    u64 total = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (auto& t : tests) {
        Position pos;
        pos.setFen(t.fen);
        u64 n = perft(pos, t.depth);
        total += n;
        bool ok = n == t.nodes;
        allOk &= ok;
        printf("%s  perft %d: %llu %s\n", ok ? "ok  " : "FAIL", t.depth, (unsigned long long)n, ok ? "" : t.fen);
    }
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("perft: %llu nodes in %.2fs = %.1fM nps\n", (unsigned long long)total, secs, total / secs / 1e6);
    for (auto& t : tests) {
        Position pos;
        pos.setFen(t.fen);
        allOk &= fuzz(pos, 2);
    }
    printf("fuzzed isPseudoLegal on %llu positions\n", (unsigned long long)checked);
    printf(allOk ? "ALL PASSED\n" : "FAILURES\n");
    return allOk ? 0 : 1;
}
