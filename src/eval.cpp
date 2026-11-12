// Hand-crafted evaluation. Only used when no network is loaded: it bootstraps
// the first generation of self-play training data.
#include "search.h"
#include <algorithm>
#include <cstdlib>

static int PstMg[6][64], PstEg[6][64];
static const int MatMg[6] = {85, 330, 350, 480, 1000, 0};
static const int MatEg[6] = {110, 300, 320, 540, 980, 0};
static const int PhaseW[6] = {0, 1, 1, 2, 4, 0};
static const int MobMg[6] = {0, 5, 5, 3, 2, 0}, MobEg[6] = {0, 5, 5, 5, 5, 0};
static const int PassedMg[8] = {0, 0, 5, 10, 25, 50, 90, 0}, PassedEg[8] = {0, 10, 15, 35, 65, 110, 170, 0};
static Bitboard PassedMask[2][64], FileMask[8], AdjFiles[8];

void initHCE() {
    for (int s = 0; s < 64; s++) {
        int r = rankOf(s), f = fileOf(s);
        int cf = std::min(f, 7 - f), cr = std::min(r, 7 - r); // 0..3 distance from edge
        int centre = cf + cr;                                 // 0..6
        PstMg[PAWN][s] = (r >= 2 && r <= 4 ? cf * 6 : 0) + (r - 1) * 3 + ((f == 3 || f == 4) && r == 1 ? -15 : 0);
        PstEg[PAWN][s] = (r - 1) * 8;
        PstMg[KNIGHT][s] = centre * 10 - 30;
        PstEg[KNIGHT][s] = centre * 6 - 18;
        PstMg[BISHOP][s] = centre * 5 - 12 + (r == 0 ? -8 : 0);
        PstEg[BISHOP][s] = centre * 4 - 10;
        PstMg[ROOK][s] = cf * 4 - 6 + (r == 6 ? 15 : 0);
        PstEg[ROOK][s] = r == 6 ? 8 : 0;
        PstMg[QUEEN][s] = centre * 2 - 6 + (r == 0 ? -4 : 0);
        PstEg[QUEEN][s] = centre * 5 - 15;
        PstMg[KING][s] = (r == 0 ? 20 : r == 1 ? 0 : -40) + (cf == 0 ? 20 : cf == 1 ? 25 : cf == 2 ? 5 : -15) * (r < 2);
        PstEg[KING][s] = centre * 12 - 36;
    }
    for (int f = 0; f < 8; f++) {
        FileMask[f] = FILE_A << f;
        AdjFiles[f] = (f > 0 ? FILE_A << (f - 1) : 0) | (f < 7 ? FILE_A << (f + 1) : 0);
    }
    for (int s = 0; s < 64; s++) {
        Bitboard span = FileMask[fileOf(s)] | AdjFiles[fileOf(s)];
        int r = rankOf(s);
        PassedMask[WHITE][s] = r < 7 ? span & (~0ULL << (8 * (r + 1))) : 0;
        PassedMask[BLACK][s] = r > 0 ? span & (~0ULL >> (8 * (8 - r))) : 0;
    }
}

int evaluateHCE(const Position& pos) {
    int mg = 0, eg = 0, phase = 0;
    const Bitboard occ = pos.occ();
    for (int c = 0; c < 2; c++) {
        int m = 0, e = 0;
        const Bitboard own = pos.colors[c], pawns = pos.bb(c, PAWN), oppPawns = pos.bb(c ^ 1, PAWN);
        for (int pt = PAWN; pt <= KING; pt++)
            for (Bitboard b = pos.bb(c, pt); b;) {
                int s = poplsb(b), rs = c == WHITE ? s : s ^ 56;
                m += MatMg[pt] + PstMg[pt][rs];
                e += MatEg[pt] + PstEg[pt][rs];
                phase += PhaseW[pt];
                if (pt == PAWN) {
                    if (!(PassedMask[c][s] & oppPawns)) {
                        m += PassedMg[rankOf(rs)];
                        e += PassedEg[rankOf(rs)];
                    }
                    if (!(AdjFiles[fileOf(s)] & pawns)) m -= 12, e -= 10;
                    if (moreThanOne(FileMask[fileOf(s)] & pawns)) m -= 6, e -= 10;
                } else if (pt != KING) {
                    int mob = popcount(attacksOf(pt, s, occ) & ~own);
                    m += MobMg[pt] * mob;
                    e += MobEg[pt] * mob;
                    if (pt == ROOK && !(FileMask[fileOf(s)] & pawns)) m += (FileMask[fileOf(s)] & oppPawns) ? 10 : 22;
                }
            }
        if (moreThanOne(pos.bb(c, BISHOP))) m += 30, e += 45;
        int ksq = pos.kingSq(c);
        m += 12 * popcount(KingAttacks[ksq] & pawns);
        Bitboard zone = KingAttacks[ksq] | sqbb(ksq);
        int danger = 0;
        for (Bitboard b = pos.colors[c ^ 1] & ~pos.pieces[PAWN] & ~pos.pieces[KING]; b;) {
            int s = poplsb(b), pt = ptOf(pos.board[s]);
            if (attacksOf(pt, s, occ) & zone) danger += pt == QUEEN ? 5 : pt == ROOK ? 3 : 2;
        }
        m -= danger * danger;
        mg += c == WHITE ? m : -m;
        eg += c == WHITE ? e : -e;
    }
    phase = std::min(phase, 24);
    int v = (mg * phase + eg * (24 - phase)) / 24;
    return (pos.stm == WHITE ? v : -v) + 12;
}
