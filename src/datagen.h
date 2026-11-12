#pragma once
#include "position.h"
#include <string>

// One training position, 32 bytes. Pieces are listed as 4-bit codes
// (colour * 6 + type) in order of the set bits of `occ`.
struct PackedPos {
    u64 occ;
    u8 pcs[16];
    i16 score; // centipawns, white-relative
    u8 result; // 0 = black win, 1 = draw, 2 = white win
    u8 stm;
    u8 pad[4];
};
static_assert(sizeof(PackedPos) == 32);

inline PackedPos packPosition(const Position& pos, int whiteScore) {
    PackedPos p{};
    p.occ = pos.occ();
    int i = 0;
    for (Bitboard b = p.occ; b; i++) {
        int s = poplsb(b);
        p.pcs[i / 2] |= u8(pos.board[s] << (4 * (i & 1)));
    }
    p.score = i16(whiteScore < -32000 ? -32000 : whiteScore > 32000 ? 32000 : whiteScore);
    p.stm = u8(pos.stm);
    return p;
}

void runDatagen(int threads, u64 positions, u64 nodes, const std::string& out, u64 seed);
void runGenFens(int count, u64 seed, const std::string& out);
