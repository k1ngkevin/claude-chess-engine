#pragma once
#include "nnue_arch.h"
#include "position.h"

#ifndef NNUE_HIDDEN
#define NNUE_HIDDEN 256
#endif
#ifndef NNUE_LAYOUT
#define NNUE_LAYOUT 0
#endif

namespace NNUE {

constexpr int HIDDEN = NNUE_HIDDEN;
constexpr Layout LAYOUT = makeLayout(NNUE_LAYOUT);
constexpr int INPUTS = LAYOUT.buckets * 768;

struct Net {
    alignas(64) i16 ftW[INPUTS * HIDDEN];
    alignas(64) i16 ftB[HIDDEN];
    alignas(64) i16 outW[OUTPUT_BUCKETS][2 * HIDDEN];
    i16 outB[OUTPUT_BUCKETS];
};

extern bool loaded;
bool loadFile(const char* path);
bool loadEmbedded();

struct Accumulator {
    alignas(64) i16 v[2][HIDDEN];
    bool computed[2];
    bool needsRefresh[2]; // the move leading here moved that side's king across a bucket/mirror boundary
    int nAdd, nSub;
    int add[2], sub[2]; // pc << 6 | sq
};

struct FinnyEntry {
    alignas(64) i16 v[HIDDEN];
    Bitboard bb[12];
};

struct State {
    Accumulator acc[MAX_PLY + 4];
    FinnyEntry finny[2][2][LAYOUT.buckets];

    void reset();                                       // call when the root position changes
    void push(int ply, const Position& before, Move m); // acc[ply] becomes the position after m
    void pushNull(int ply);
    int evaluate(const Position& pos, int ply);         // side-to-move relative, centipawns

private:
    void refresh(const Position& pos, int ply, int persp);
    void update(const Position& pos, int ply, int persp);
};

} // namespace NNUE
