#pragma once
#include <cstdint>
#include <string>

using u8 = uint8_t;
using u16 = uint16_t;
using i16 = int16_t;
using i32 = int32_t;
using u32 = uint32_t;
using u64 = uint64_t;
using i64 = int64_t;
using Bitboard = u64;
using Key = u64;

enum { WHITE, BLACK };
enum { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, NO_PT };
constexpr int NO_PIECE = 12; // pieces are color * 6 + type
constexpr int SQ_NONE = 64;

inline int makePiece(int c, int pt) { return c * 6 + pt; }
inline int ptOf(int pc) { return pc % 6; }
inline int colorOf(int pc) { return pc >= 6; }
inline int fileOf(int s) { return s & 7; }
inline int rankOf(int s) { return s >> 3; }
inline int relRank(int c, int s) { return c == WHITE ? s >> 3 : 7 - (s >> 3); }

// Move: from (6 bits) | to (6 bits) | flag (4 bits)
using Move = u16;
enum : int {
    QUIET = 0, DOUBLE_PUSH = 1, KING_CASTLE = 2, QUEEN_CASTLE = 3,
    CAPTURE = 4, EP = 5, PROMO = 8, PROMO_CAPTURE = 12
};
constexpr Move MOVE_NONE = 0;
constexpr Move MOVE_NULL = 65;

inline Move mkMove(int from, int to, int flag) { return Move(from | (to << 6) | (flag << 12)); }
inline int fromSq(Move m) { return m & 63; }
inline int toSq(Move m) { return (m >> 6) & 63; }
inline int flagOf(Move m) { return m >> 12; }
inline bool isCapture(Move m) { return (m >> 12) & 4; }
inline bool isPromo(Move m) { return (m >> 12) & 8; }
inline int promoPt(Move m) { return ((m >> 12) & 3) + KNIGHT; }
inline bool isCastle(Move m) { return flagOf(m) == KING_CASTLE || flagOf(m) == QUEEN_CASTLE; }
inline bool isNoisy(Move m) { return (m >> 12) & 12; } // capture or promotion

constexpr int MAX_PLY = 200;
constexpr int SCORE_MATE = 32000;
constexpr int SCORE_INF = 32001;
constexpr int SCORE_NONE = 32002;
constexpr int MATE_BOUND = SCORE_MATE - 2 * MAX_PLY; // |score| >= this means forced mate

std::string moveToStr(Move m);
