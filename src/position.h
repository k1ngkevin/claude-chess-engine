#pragma once
#include "bitboard.h"
#include <string>

struct MoveList {
    Move moves[256];
    int n = 0;
    void add(Move m) { moves[n++] = m; }
};

enum { GEN_NOISY = 1, GEN_QUIET = 2, GEN_ALL = 3 };

extern const char* const STARTPOS;

struct Position {
    Bitboard pieces[6];
    Bitboard colors[2];
    u8 board[64];
    Key key, pawnKey, nonPawnKey[2], minorKey, majorKey;
    Bitboard checkers; // enemy pieces giving check to the side to move
    Bitboard pinned;   // side-to-move pieces pinned to their own king
    int stm, ep, castling, rule50, gamePly;

    Bitboard occ() const { return colors[0] | colors[1]; }
    Bitboard bb(int c, int pt) const { return pieces[pt] & colors[c]; }
    int kingSq(int c) const { return lsb(pieces[KING] & colors[c]); }
    bool hasNonPawn(int c) const { return colors[c] & ~(pieces[PAWN] | pieces[KING]); }

    void setFen(const std::string& fen);
    std::string fen() const;

    Bitboard attackersTo(int sq, Bitboard occ) const;
    bool attacked(int sq, int by) const { return attackersTo(sq, occ()) & colors[by]; }
    Bitboard threatsBy(int c) const; // all squares attacked by color c

    void makeMove(Move m);
    void makeNull();
    bool isPseudoLegal(Move m) const;
    bool isLegal(Move m) const; // requires a pseudo-legal move
    bool see(Move m, int threshold) const;
    bool insufficientMaterial() const;
    int capturedPt(Move m) const { return flagOf(m) == EP ? PAWN : board[toSq(m)] == NO_PIECE ? NO_PT : ptOf(board[toSq(m)]); }

private:
    void put(int pc, int s);
    void remove(int s);
    void updateCheckInfo();
};

void initZobrist();
void generate(const Position& pos, MoveList& list, int type);
void generateLegal(const Position& pos, MoveList& list);
Move parseMove(const Position& pos, const std::string& s);
u64 perft(const Position& pos, int depth);

extern const int SeeValue[7];
