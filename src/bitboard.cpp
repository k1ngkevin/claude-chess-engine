#include "bitboard.h"

Magic RookMagics[64], BishopMagics[64];
Bitboard KnightAttacks[64], KingAttacks[64], PawnAttacks[2][64];
Bitboard BetweenBB[64][64], LineBB[64][64];

static Bitboard RookTable[102400], BishopTable[5248];

static Bitboard slide(int sq, Bitboard occ, bool rook) {
    static const int D[2][4][2] = {{{1, 1}, {1, -1}, {-1, 1}, {-1, -1}}, {{1, 0}, {-1, 0}, {0, 1}, {0, -1}}};
    Bitboard a = 0;
    for (int d = 0; d < 4; d++) {
        int r = rankOf(sq) + D[rook][d][0], f = fileOf(sq) + D[rook][d][1];
        while (r >= 0 && r < 8 && f >= 0 && f < 8) {
            Bitboard b = sqbb(r * 8 + f);
            a |= b;
            if (occ & b) break;
            r += D[rook][d][0];
            f += D[rook][d][1];
        }
    }
    return a;
}

static u64 rngState = 0x2545F4914F6CDD1DULL;
static u64 rnd() {
    rngState ^= rngState >> 12;
    rngState ^= rngState << 25;
    rngState ^= rngState >> 27;
    return rngState * 0x2545F4914F6CDD1DULL;
}

// Finds a collision-free magic multiplier for every square by trial.
static void initMagics(Magic* magics, Bitboard* table, bool rook) {
    static Bitboard occs[4096], refs[4096];
    static int epoch[4096];
    int cnt = 0, size = 0;
    for (int s = 0; s < 64; s++) {
        Bitboard edges = ((RANK_1 | RANK_8) & ~(RANK_1 << (8 * rankOf(s)))) | ((FILE_A | FILE_H) & ~(FILE_A << fileOf(s)));
        Magic& m = magics[s];
        m.mask = slide(s, 0, rook) & ~edges;
        m.shift = 64 - popcount(m.mask);
        m.att = s == 0 ? table : magics[s - 1].att + size;
        size = 0;
        Bitboard b = 0;
        do {
            occs[size] = b;
            refs[size] = slide(s, b, rook);
            size++;
            b = (b - m.mask) & m.mask;
        } while (b);
        for (;;) {
            m.magic = rnd() & rnd() & rnd();
            if (popcount((m.magic * m.mask) >> 56) < 6) continue;
            ++cnt;
            bool ok = true;
            for (int i = 0; i < size && ok; i++) {
                unsigned idx = unsigned((occs[i] * m.magic) >> m.shift);
                if (epoch[idx] < cnt) {
                    epoch[idx] = cnt;
                    m.att[idx] = refs[i];
                } else if (m.att[idx] != refs[i])
                    ok = false;
            }
            if (ok) break;
        }
    }
}

void initBitboards() {
    for (int s = 0; s < 64; s++) {
        int r = rankOf(s), f = fileOf(s);
        static const int KN[8][2] = {{1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}};
        for (auto& d : KN) {
            int rr = r + d[0], ff = f + d[1];
            if (rr >= 0 && rr < 8 && ff >= 0 && ff < 8) KnightAttacks[s] |= sqbb(rr * 8 + ff);
        }
        for (int dr = -1; dr <= 1; dr++)
            for (int df = -1; df <= 1; df++) {
                int rr = r + dr, ff = f + df;
                if ((dr || df) && rr >= 0 && rr < 8 && ff >= 0 && ff < 8) KingAttacks[s] |= sqbb(rr * 8 + ff);
            }
        for (int df = -1; df <= 1; df += 2) {
            int ff = f + df;
            if (ff < 0 || ff > 7) continue;
            if (r < 7) PawnAttacks[WHITE][s] |= sqbb((r + 1) * 8 + ff);
            if (r > 0) PawnAttacks[BLACK][s] |= sqbb((r - 1) * 8 + ff);
        }
    }
    initMagics(RookMagics, RookTable, true);
    initMagics(BishopMagics, BishopTable, false);
    for (int a = 0; a < 64; a++)
        for (int b = 0; b < 64; b++) {
            if (a == b) continue;
            if (rookAttacks(a, 0) & sqbb(b)) {
                LineBB[a][b] = (rookAttacks(a, 0) & rookAttacks(b, 0)) | sqbb(a) | sqbb(b);
                BetweenBB[a][b] = rookAttacks(a, sqbb(b)) & rookAttacks(b, sqbb(a));
            } else if (bishopAttacks(a, 0) & sqbb(b)) {
                LineBB[a][b] = (bishopAttacks(a, 0) & bishopAttacks(b, 0)) | sqbb(a) | sqbb(b);
                BetweenBB[a][b] = bishopAttacks(a, sqbb(b)) & bishopAttacks(b, sqbb(a));
            }
        }
}
