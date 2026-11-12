#include "position.h"
#include <cstring>
#include <sstream>

const char* const STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const int SeeValue[7] = {100, 320, 330, 500, 950, 20000, 0};

static Key ZPsq[12][64], ZEp[8], ZCastle[16], ZSide;
static int CastleMask[64];
static const char PieceChars[] = "PNBRQKpnbrqk";

void initZobrist() {
    u64 s = 0x9E3779B97F4A7C15ULL;
    auto next = [&]() {
        u64 z = (s += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    };
    for (auto& a : ZPsq)
        for (auto& k : a) k = next();
    for (auto& k : ZEp) k = next();
    for (auto& k : ZCastle) k = next();
    ZCastle[0] = 0;
    ZSide = next();
    for (int& m : CastleMask) m = 15;
    CastleMask[0] = 15 & ~2;
    CastleMask[4] = 15 & ~3;
    CastleMask[7] = 15 & ~1;
    CastleMask[56] = 15 & ~8;
    CastleMask[60] = 15 & ~12;
    CastleMask[63] = 15 & ~4;
}

std::string moveToStr(Move m) {
    if (m == MOVE_NONE || m == MOVE_NULL) return "0000";
    std::string s;
    s += char('a' + fileOf(fromSq(m)));
    s += char('1' + rankOf(fromSq(m)));
    s += char('a' + fileOf(toSq(m)));
    s += char('1' + rankOf(toSq(m)));
    if (isPromo(m)) s += "nbrq"[promoPt(m) - KNIGHT];
    return s;
}

inline void Position::put(int pc, int s) {
    board[s] = u8(pc);
    Bitboard b = sqbb(s);
    int pt = pc % 6, c = pc >= 6;
    pieces[pt] |= b;
    colors[c] |= b;
    Key k = ZPsq[pc][s];
    key ^= k;
    if (pt == PAWN)
        pawnKey ^= k;
    else {
        nonPawnKey[c] ^= k;
        if (pt == KNIGHT || pt == BISHOP || pt == KING) minorKey ^= k;
        if (pt == ROOK || pt == QUEEN || pt == KING) majorKey ^= k;
    }
}

inline void Position::remove(int s) {
    int pc = board[s];
    board[s] = NO_PIECE;
    Bitboard b = sqbb(s);
    int pt = pc % 6, c = pc >= 6;
    pieces[pt] ^= b;
    colors[c] ^= b;
    Key k = ZPsq[pc][s];
    key ^= k;
    if (pt == PAWN)
        pawnKey ^= k;
    else {
        nonPawnKey[c] ^= k;
        if (pt == KNIGHT || pt == BISHOP || pt == KING) minorKey ^= k;
        if (pt == ROOK || pt == QUEEN || pt == KING) majorKey ^= k;
    }
}

void Position::setFen(const std::string& fen) {
    memset(this, 0, sizeof(*this));
    memset(board, NO_PIECE, sizeof(board));
    std::istringstream ss(fen);
    std::string brd, side, castle, eps;
    int hm = 0, fm = 1;
    ss >> brd >> side >> castle >> eps;
    if (!(ss >> hm)) hm = 0;
    if (!(ss >> fm)) fm = 1;
    int s = 56;
    for (char ch : brd) {
        if (ch == '/')
            s -= 16;
        else if (ch >= '1' && ch <= '8')
            s += ch - '0';
        else {
            const char* p = strchr(PieceChars, ch);
            if (p && s >= 0 && s < 64) put(int(p - PieceChars), s);
            s++;
        }
    }
    stm = side == "b" ? BLACK : WHITE;
    if (stm == BLACK) key ^= ZSide;
    for (char ch : castle) {
        if (ch == 'K') castling |= 1;
        if (ch == 'Q') castling |= 2;
        if (ch == 'k') castling |= 4;
        if (ch == 'q') castling |= 8;
    }
    key ^= ZCastle[castling];
    ep = SQ_NONE;
    if (eps.size() == 2 && eps[0] >= 'a' && eps[0] <= 'h') {
        int e = (eps[0] - 'a') + 8 * (eps[1] - '1');
        if (PawnAttacks[stm ^ 1][e] & bb(stm, PAWN)) {
            ep = e;
            key ^= ZEp[fileOf(e)];
        }
    }
    rule50 = hm;
    gamePly = (fm - 1) * 2 + stm;
    updateCheckInfo();
}

std::string Position::fen() const {
    std::string s;
    for (int r = 7; r >= 0; r--) {
        int empty = 0;
        for (int f = 0; f < 8; f++) {
            int pc = board[r * 8 + f];
            if (pc == NO_PIECE)
                empty++;
            else {
                if (empty) s += char('0' + empty);
                empty = 0;
                s += PieceChars[pc];
            }
        }
        if (empty) s += char('0' + empty);
        if (r) s += '/';
    }
    s += stm == WHITE ? " w " : " b ";
    if (!castling) s += '-';
    if (castling & 1) s += 'K';
    if (castling & 2) s += 'Q';
    if (castling & 4) s += 'k';
    if (castling & 8) s += 'q';
    s += ' ';
    if (ep == SQ_NONE)
        s += '-';
    else {
        s += char('a' + fileOf(ep));
        s += char('1' + rankOf(ep));
    }
    s += ' ' + std::to_string(rule50) + ' ' + std::to_string(gamePly / 2 + 1);
    return s;
}

Bitboard Position::attackersTo(int sq, Bitboard o) const {
    return (PawnAttacks[BLACK][sq] & pieces[PAWN] & colors[WHITE]) | (PawnAttacks[WHITE][sq] & pieces[PAWN] & colors[BLACK]) |
           (KnightAttacks[sq] & pieces[KNIGHT]) | (KingAttacks[sq] & pieces[KING]) |
           (bishopAttacks(sq, o) & (pieces[BISHOP] | pieces[QUEEN])) | (rookAttacks(sq, o) & (pieces[ROOK] | pieces[QUEEN]));
}

Bitboard Position::threatsBy(int c) const {
    Bitboard o = occ(), t = 0, p = bb(c, PAWN);
    t = c == WHITE ? ((p & ~FILE_A) << 7) | ((p & ~FILE_H) << 9) : ((p & ~FILE_A) >> 9) | ((p & ~FILE_H) >> 7);
    for (Bitboard b = bb(c, KNIGHT); b;) t |= KnightAttacks[poplsb(b)];
    for (Bitboard b = (pieces[BISHOP] | pieces[QUEEN]) & colors[c]; b;) t |= bishopAttacks(poplsb(b), o);
    for (Bitboard b = (pieces[ROOK] | pieces[QUEEN]) & colors[c]; b;) t |= rookAttacks(poplsb(b), o);
    return t | KingAttacks[kingSq(c)];
}

void Position::updateCheckInfo() {
    int us = stm, them = us ^ 1, ksq = kingSq(us);
    Bitboard o = occ();
    checkers = attackersTo(ksq, o) & colors[them];
    pinned = 0;
    Bitboard snipers = ((rookAttacks(ksq, 0) & (pieces[ROOK] | pieces[QUEEN])) | (bishopAttacks(ksq, 0) & (pieces[BISHOP] | pieces[QUEEN]))) & colors[them];
    while (snipers) {
        Bitboard b = BetweenBB[ksq][poplsb(snipers)] & o;
        if (b && !moreThanOne(b)) pinned |= b & colors[us];
    }
}

void Position::makeMove(Move m) {
    const int from = fromSq(m), to = toSq(m), flag = flagOf(m);
    const int us = stm, them = us ^ 1;
    const int pc = board[from];
    rule50++;
    gamePly++;
    if (ep != SQ_NONE) {
        key ^= ZEp[fileOf(ep)];
        ep = SQ_NONE;
    }
    if (flag == EP) {
        remove(us == WHITE ? to - 8 : to + 8);
        rule50 = 0;
    } else if (isCapture(m)) {
        remove(to);
        rule50 = 0;
    }
    remove(from);
    if (isPromo(m))
        put(makePiece(us, promoPt(m)), to);
    else
        put(pc, to);
    if (flag == KING_CASTLE) {
        int r = to + 1;
        int rook = board[r];
        remove(r);
        put(rook, to - 1);
    } else if (flag == QUEEN_CASTLE) {
        int r = to - 2;
        int rook = board[r];
        remove(r);
        put(rook, to + 1);
    }
    if (pc % 6 == PAWN) {
        rule50 = 0;
        if (flag == DOUBLE_PUSH) {
            int e = (from + to) / 2;
            if (PawnAttacks[us][e] & bb(them, PAWN)) {
                ep = e;
                key ^= ZEp[fileOf(e)];
            }
        }
    }
    key ^= ZCastle[castling];
    castling &= CastleMask[from] & CastleMask[to];
    key ^= ZCastle[castling];
    stm = them;
    key ^= ZSide;
    updateCheckInfo();
}

void Position::makeNull() {
    rule50++;
    gamePly++;
    if (ep != SQ_NONE) {
        key ^= ZEp[fileOf(ep)];
        ep = SQ_NONE;
    }
    stm ^= 1;
    key ^= ZSide;
    updateCheckInfo();
}

bool Position::insufficientMaterial() const {
    if (pieces[PAWN] | pieces[ROOK] | pieces[QUEEN]) return false;
    return popcount(occ()) <= 3;
}

bool Position::isLegal(Move m) const {
    const int us = stm, them = us ^ 1, from = fromSq(m), to = toSq(m), ksq = kingSq(us);
    if (flagOf(m) == EP) {
        int capsq = us == WHITE ? to - 8 : to + 8;
        Bitboard o = (occ() ^ sqbb(from) ^ sqbb(capsq)) | sqbb(to);
        return !(rookAttacks(ksq, o) & (pieces[ROOK] | pieces[QUEEN]) & colors[them]) &&
               !(bishopAttacks(ksq, o) & (pieces[BISHOP] | pieces[QUEEN]) & colors[them]);
    }
    if (from == ksq) {
        if (isCastle(m)) return true; // path safety is verified at generation
        return !(attackersTo(to, occ() ^ sqbb(ksq)) & colors[them]);
    }
    return !(pinned & sqbb(from)) || (LineBB[from][to] & sqbb(ksq));
}

bool Position::isPseudoLegal(Move m) const {
    if (m == MOVE_NONE) return false;
    const int from = fromSq(m), to = toSq(m), flag = flagOf(m), us = stm, them = us ^ 1;
    const int pc = board[from];
    if (pc == NO_PIECE || colorOf(pc) != us) return false;
    const int pt = pc % 6;
    const Bitboard o = occ(), toBB = sqbb(to);
    if (colors[us] & toBB) return false;
    if (flag == 6 || flag == 7) return false;
    const bool cap = board[to] != NO_PIECE;
    if (flag == KING_CASTLE || flag == QUEEN_CASTLE) {
        if (pt != KING || checkers) return false;
        int base = us == WHITE ? 0 : 56;
        if (from != base + 4) return false;
        if (flag == KING_CASTLE)
            return to == base + 6 && (castling & (us == WHITE ? 1 : 4)) && !(o & (0x60ULL << base)) && !attacked(base + 5, them) && !attacked(base + 6, them);
        return to == base + 2 && (castling & (us == WHITE ? 2 : 8)) && !(o & (0x0EULL << base)) && !attacked(base + 3, them) && !attacked(base + 2, them);
    }
    if (pt == PAWN) {
        const int up = us == WHITE ? 8 : -8;
        if (isPromo(m) != (relRank(us, to) == 7)) return false;
        if (flag == EP) {
            if (to != ep || !(PawnAttacks[us][from] & toBB)) return false;
        } else if (isCapture(m)) {
            if (!cap || !(PawnAttacks[us][from] & toBB)) return false;
        } else if (flag == DOUBLE_PUSH) {
            if (relRank(us, from) != 1 || to != from + 2 * up || (o & (toBB | sqbb(from + up)))) return false;
        } else {
            if (to != from + up || cap) return false;
        }
    } else {
        if (flag != QUIET && flag != CAPTURE) return false;
        if ((flag == CAPTURE) != cap) return false;
        if (!(attacksOf(pt, from, o) & toBB)) return false;
    }
    if (checkers && pt != KING) {
        if (moreThanOne(checkers)) return false;
        Bitboard target = BetweenBB[kingSq(us)][lsb(checkers)] | checkers;
        if (flag == EP) {
            int capsq = us == WHITE ? to - 8 : to + 8;
            if (!(checkers & sqbb(capsq)) && !(target & toBB)) return false;
        } else if (!(target & toBB))
            return false;
    }
    return true;
}

// Static exchange evaluation: does the exchange started by m win at least `threshold`?
bool Position::see(Move m, int threshold) const {
    if (isCastle(m)) return threshold <= 0;
    const int from = fromSq(m), to = toSq(m);
    int value = SeeValue[capturedPt(m)] - threshold;
    int next = ptOf(board[from]);
    if (isPromo(m)) {
        value += SeeValue[promoPt(m)] - SeeValue[PAWN];
        next = promoPt(m);
    }
    if (value < 0) return false;
    value -= SeeValue[next];
    if (value >= 0) return true;
    Bitboard o = (occ() ^ sqbb(from)) | sqbb(to);
    if (flagOf(m) == EP) o ^= sqbb(stm == WHITE ? to - 8 : to + 8);
    Bitboard attackers = attackersTo(to, o) & o;
    const Bitboard diag = pieces[BISHOP] | pieces[QUEEN], orth = pieces[ROOK] | pieces[QUEEN];
    int side = stm ^ 1;
    for (;;) {
        Bitboard mine = attackers & colors[side];
        if (!mine) break;
        int pt = PAWN;
        while (!(mine & pieces[pt])) pt++;
        side ^= 1;
        value = -value - 1 - SeeValue[pt];
        if (value >= 0) {
            if (pt == KING && (attackers & colors[side])) side ^= 1;
            break;
        }
        o ^= sqbb(lsb(mine & pieces[pt]));
        if (pt == PAWN || pt == BISHOP || pt == QUEEN) attackers |= bishopAttacks(to, o) & diag;
        if (pt == ROOK || pt == QUEEN) attackers |= rookAttacks(to, o) & orth;
        attackers &= o;
    }
    return side != stm;
}

// ---------------------------------------------------------------- move generation

void generate(const Position& pos, MoveList& ml, int type) {
    const int us = pos.stm, them = us ^ 1, ksq = pos.kingSq(us);
    const Bitboard o = pos.occ(), theirs = pos.colors[them], checkers = pos.checkers;
    const bool noisy = type & GEN_NOISY, quiet = type & GEN_QUIET;

    Bitboard katt = KingAttacks[ksq];
    if (noisy)
        for (Bitboard b = katt & theirs; b;) ml.add(mkMove(ksq, poplsb(b), CAPTURE));
    if (quiet)
        for (Bitboard b = katt & ~o; b;) ml.add(mkMove(ksq, poplsb(b), QUIET));
    if (moreThanOne(checkers)) return;

    const Bitboard target = checkers ? BetweenBB[ksq][lsb(checkers)] | checkers : ~0ULL;
    const Bitboard T = ((noisy ? theirs : 0) | (quiet ? ~o : 0)) & target;

    const Bitboard pawns = pos.bb(us, PAWN);
    const Bitboard r7 = us == WHITE ? RANK_7 : RANK_2, r3 = us == WHITE ? RANK_3 : RANK_6;
    const int up = us == WHITE ? 8 : -8, upL = us == WHITE ? 7 : -9, upR = us == WHITE ? 9 : -7;
    auto shUp = [&](Bitboard b) { return us == WHITE ? b << 8 : b >> 8; };
    auto shL = [&](Bitboard b) { return us == WHITE ? (b & ~FILE_A) << 7 : (b & ~FILE_A) >> 9; };
    auto shR = [&](Bitboard b) { return us == WHITE ? (b & ~FILE_H) << 9 : (b & ~FILE_H) >> 7; };
    const Bitboard promoP = pawns & r7, nonP = pawns & ~r7;

    if (quiet) {
        Bitboard s1 = shUp(nonP) & ~o;
        Bitboard s2 = shUp(s1 & r3) & ~o & target;
        s1 &= target;
        while (s1) {
            int to = poplsb(s1);
            ml.add(mkMove(to - up, to, QUIET));
        }
        while (s2) {
            int to = poplsb(s2);
            ml.add(mkMove(to - 2 * up, to, DOUBLE_PUSH));
        }
    }
    if (noisy) {
        for (Bitboard b = shL(nonP) & theirs & target; b;) {
            int to = poplsb(b);
            ml.add(mkMove(to - upL, to, CAPTURE));
        }
        for (Bitboard b = shR(nonP) & theirs & target; b;) {
            int to = poplsb(b);
            ml.add(mkMove(to - upR, to, CAPTURE));
        }
        if (promoP) {
            for (Bitboard b = shUp(promoP) & ~o & target; b;) {
                int to = poplsb(b);
                for (int p = 3; p >= 0; p--) ml.add(mkMove(to - up, to, PROMO + p));
            }
            for (Bitboard b = shL(promoP) & theirs & target; b;) {
                int to = poplsb(b);
                for (int p = 3; p >= 0; p--) ml.add(mkMove(to - upL, to, PROMO_CAPTURE + p));
            }
            for (Bitboard b = shR(promoP) & theirs & target; b;) {
                int to = poplsb(b);
                for (int p = 3; p >= 0; p--) ml.add(mkMove(to - upR, to, PROMO_CAPTURE + p));
            }
        }
        if (pos.ep != SQ_NONE) {
            int capsq = pos.ep - up;
            if (!checkers || (checkers & sqbb(capsq)) || (target & sqbb(pos.ep)))
                for (Bitboard b = PawnAttacks[them][pos.ep] & pawns; b;) ml.add(mkMove(poplsb(b), pos.ep, EP));
        }
    }

    for (int pt = KNIGHT; pt <= QUEEN; pt++)
        for (Bitboard b = pos.bb(us, pt); b;) {
            int from = poplsb(b);
            for (Bitboard a = attacksOf(pt, from, o) & T; a;) {
                int to = poplsb(a);
                ml.add(mkMove(from, to, (theirs >> to) & 1 ? CAPTURE : QUIET));
            }
        }

    if (quiet && !checkers) {
        int base = us == WHITE ? 0 : 56;
        if ((pos.castling & (us == WHITE ? 1 : 4)) && !(o & (0x60ULL << base)) && !pos.attacked(base + 5, them) && !pos.attacked(base + 6, them))
            ml.add(mkMove(base + 4, base + 6, KING_CASTLE));
        if ((pos.castling & (us == WHITE ? 2 : 8)) && !(o & (0x0EULL << base)) && !pos.attacked(base + 3, them) && !pos.attacked(base + 2, them))
            ml.add(mkMove(base + 4, base + 2, QUEEN_CASTLE));
    }
}

void generateLegal(const Position& pos, MoveList& list) {
    MoveList all;
    generate(pos, all, GEN_ALL);
    for (int i = 0; i < all.n; i++)
        if (pos.isLegal(all.moves[i])) list.add(all.moves[i]);
}

Move parseMove(const Position& pos, const std::string& s) {
    MoveList ml;
    generateLegal(pos, ml);
    for (int i = 0; i < ml.n; i++)
        if (moveToStr(ml.moves[i]) == s) return ml.moves[i];
    return MOVE_NONE;
}

u64 perft(const Position& pos, int depth) {
    MoveList ml;
    generate(pos, ml, GEN_ALL);
    u64 n = 0;
    for (int i = 0; i < ml.n; i++) {
        if (!pos.isLegal(ml.moves[i])) continue;
        if (depth == 1) {
            n++;
            continue;
        }
        Position next = pos;
        next.makeMove(ml.moves[i]);
        n += perft(next, depth - 1);
    }
    return n;
}
