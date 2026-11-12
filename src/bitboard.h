#pragma once
#include "types.h"

inline int popcount(Bitboard b) { return __builtin_popcountll(b); }
inline int lsb(Bitboard b) { return __builtin_ctzll(b); }
inline int poplsb(Bitboard& b) { int s = lsb(b); b &= b - 1; return s; }
inline Bitboard sqbb(int s) { return 1ULL << s; }
inline bool moreThanOne(Bitboard b) { return b & (b - 1); }

constexpr Bitboard FILE_A = 0x0101010101010101ULL;
constexpr Bitboard FILE_H = FILE_A << 7;
constexpr Bitboard RANK_1 = 0xFFULL;
constexpr Bitboard RANK_2 = RANK_1 << 8;
constexpr Bitboard RANK_3 = RANK_1 << 16;
constexpr Bitboard RANK_6 = RANK_1 << 40;
constexpr Bitboard RANK_7 = RANK_1 << 48;
constexpr Bitboard RANK_8 = RANK_1 << 56;

struct Magic {
    Bitboard mask, magic;
    Bitboard* att;
    int shift;
};
extern Magic RookMagics[64], BishopMagics[64];
extern Bitboard KnightAttacks[64], KingAttacks[64], PawnAttacks[2][64];
extern Bitboard BetweenBB[64][64]; // squares strictly between two aligned squares
extern Bitboard LineBB[64][64];    // full line through two aligned squares

inline Bitboard rookAttacks(int s, Bitboard occ) {
    const Magic& m = RookMagics[s];
    return m.att[((occ & m.mask) * m.magic) >> m.shift];
}
inline Bitboard bishopAttacks(int s, Bitboard occ) {
    const Magic& m = BishopMagics[s];
    return m.att[((occ & m.mask) * m.magic) >> m.shift];
}
inline Bitboard attacksOf(int pt, int s, Bitboard occ) {
    switch (pt) {
    case KNIGHT: return KnightAttacks[s];
    case BISHOP: return bishopAttacks(s, occ);
    case ROOK: return rookAttacks(s, occ);
    case QUEEN: return bishopAttacks(s, occ) | rookAttacks(s, occ);
    default: return KingAttacks[s];
    }
}

void initBitboards();
