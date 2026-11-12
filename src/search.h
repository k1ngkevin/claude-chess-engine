#pragma once
#include "nnue.h"
#include "position.h"
#include "tt.h"
#include <atomic>
#include <chrono>
#include <memory>
#include <vector>

inline i64 nowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Limits {
    i64 start = 0;
    i64 softMs = -1, hardMs = -1; // -1: no time limit
    u64 softNodes = 0, hardNodes = 0;
    int depth = MAX_PLY - 1;
};

using PieceTo = i16[12][64];

struct Stack {
    Move move;
    Move killer;
    Move excluded;
    int ply;
    int staticEval;
    int movedPiece;
    int cutoffCnt;
    int reduction;
    bool ttPv;
    Bitboard threats;
    PieceTo* contHist;
    PieceTo* contCorr;
    int pvLen;
    Move pv[MAX_PLY + 2];
};

constexpr int CORR_SIZE = 16384;

struct Searcher {
    // configuration
    TT* tt = nullptr;
    std::atomic<bool>* stop = nullptr;
    bool isMain = true, silent = false;
    const std::vector<std::unique_ptr<Searcher>>* pool = nullptr; // all threads of this search, for node totals
    Limits lim;

    // root state
    Position rootPos;
    Key keys[512 + MAX_PLY + 8];
    int keyBase = 0; // index of the root position's key in keys[]

    // results
    Move bestMove = MOVE_NONE;
    int bestScore = 0, completedDepth = 0, selDepth = 0, rootDepth = 0;
    std::atomic<u64> nodes{0};
    Move rootPv[MAX_PLY + 2];
    int rootPvLen = 0;

    // heuristics
    i16 mainHist[2][2][2][4096];
    i16 captHist[12][64][7];
    i16 contHist[13][64][12][64];
    i16 pawnHist[1024][12][64];
    i16 pawnCorr[2][CORR_SIZE], nonPawnCorr[2][2][CORR_SIZE], minorCorr[2][CORR_SIZE], majorCorr[2][CORR_SIZE];
    i16 contCorr[13][64][12][64];
    u64 rootNodes[4096];
    int nmpMinPly = 0;
    Stack stack[MAX_PLY + 12];
    NNUE::State nn;

    void clearHistory();
    void setRoot(const Position& pos, const std::vector<Key>& history); // history ends with pos.key
    void think();

    int search(const Position& pos, Stack* ss, int alpha, int beta, int depth, bool cutNode);
    int qsearch(const Position& pos, Stack* ss, int alpha, int beta);
    int evaluate(const Position& pos, Stack* ss);
    int correction(const Position& pos, Stack* ss) const;
    void updateCorrection(const Position& pos, Stack* ss, int diff, int depth);
    bool isDraw(const Position& pos, int ply) const;
    bool shouldStop();
    int quietHistory(const Position& pos, Stack* ss, Move m) const;
    void updateQuiet(const Position& pos, Stack* ss, Move m, int bonus);
    void updateCont(Stack* ss, int pc, int to, int bonus);
    i16& captEntry(const Position& pos, Move m) { return captHist[pos.board[fromSq(m)]][toSq(m)][pos.capturedPt(m)]; }
    u64 nodeCount() const { return nodes.load(std::memory_order_relaxed); }
};

void initSearch();
int evaluateHCE(const Position& pos);
void initHCE();
