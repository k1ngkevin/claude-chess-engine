#include "search.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

static int LmrTable[2][64][64]; // [quiet][depth][moveCount], in 1/1024 plies

void initSearch() {
    for (int d = 1; d < 64; d++)
        for (int m = 1; m < 64; m++) {
            LmrTable[0][d][m] = int(1024 * (0.20 + std::log(d) * std::log(m) / 3.30));
            LmrTable[1][d][m] = int(1024 * (0.80 + std::log(d) * std::log(m) / 2.40));
        }
}

constexpr int HIST_MAX = 16384;
constexpr int CORR_MAX = 1024;

static inline void gravity(i16& e, int bonus, int max = HIST_MAX) {
    bonus = std::clamp(bonus, -max, max);
    e = i16(e + bonus - e * std::abs(bonus) / max);
}
static inline int scoreToTT(int s, int ply) { return s >= MATE_BOUND ? s + ply : s <= -MATE_BOUND ? s - ply : s; }
static inline int scoreFromTT(int s, int ply) {
    if (s == SCORE_NONE) return s;
    return s >= MATE_BOUND ? s - ply : s <= -MATE_BOUND ? s + ply : s;
}

void Searcher::clearHistory() {
    memset(mainHist, 0, sizeof(mainHist));
    memset(captHist, 0, sizeof(captHist));
    memset(contHist, 0, sizeof(contHist));
    memset(pawnHist, 0, sizeof(pawnHist));
    memset(pawnCorr, 0, sizeof(pawnCorr));
    memset(nonPawnCorr, 0, sizeof(nonPawnCorr));
    memset(minorCorr, 0, sizeof(minorCorr));
    memset(majorCorr, 0, sizeof(majorCorr));
    memset(contCorr, 0, sizeof(contCorr));
}

void Searcher::setRoot(const Position& pos, const std::vector<Key>& history) {
    rootPos = pos;
    int n = std::min<int>((int)history.size(), 500);
    for (int i = 0; i < n; i++) keys[i] = history[history.size() - n + i];
    if (n == 0) keys[n++] = pos.key;
    keyBase = n - 1;
}

bool Searcher::isDraw(const Position& pos, int ply) const {
    if (pos.rule50 >= 100 || pos.insufficientMaterial()) return true;
    int idx = keyBase + ply, end = std::max(0, idx - pos.rule50);
    for (int i = idx - 4; i >= end; i -= 2)
        if (keys[i] == pos.key) return true;
    return false;
}

bool Searcher::shouldStop() {
    u64 n = nodeCount();
    if ((n & 1023) == 0 && rootDepth > 1) {
        if (lim.hardNodes && n >= lim.hardNodes) stop->store(true, std::memory_order_relaxed);
        if (isMain && lim.hardMs >= 0 && nowMs() - lim.start >= lim.hardMs) stop->store(true, std::memory_order_relaxed);
    }
    return stop->load(std::memory_order_relaxed);
}

int Searcher::evaluate(const Position& pos, Stack* ss) {
    int v = NNUE::loaded ? nn.evaluate(pos, ss->ply) : evaluateHCE(pos);
    return std::clamp(v, -MATE_BOUND + 1, MATE_BOUND - 1);
}

int Searcher::correction(const Position& pos, Stack* ss) const {
    const int us = pos.stm;
    int c = 64 * pawnCorr[us][pos.pawnKey & (CORR_SIZE - 1)];
    c += 64 * nonPawnCorr[us][0][pos.nonPawnKey[0] & (CORR_SIZE - 1)];
    c += 64 * nonPawnCorr[us][1][pos.nonPawnKey[1] & (CORR_SIZE - 1)];
    c += 48 * minorCorr[us][pos.minorKey & (CORR_SIZE - 1)];
    c += 48 * majorCorr[us][pos.majorKey & (CORR_SIZE - 1)];
    if ((ss - 1)->movedPiece != NO_PIECE) c += 64 * (*(ss - 2)->contCorr)[(ss - 1)->movedPiece][toSq((ss - 1)->move)];
    return c / 1024;
}

void Searcher::updateCorrection(const Position& pos, Stack* ss, int diff, int depth) {
    const int us = pos.stm;
    const int bonus = std::clamp(diff * depth / 8, -CORR_MAX / 4, CORR_MAX / 4);
    gravity(pawnCorr[us][pos.pawnKey & (CORR_SIZE - 1)], bonus, CORR_MAX);
    gravity(nonPawnCorr[us][0][pos.nonPawnKey[0] & (CORR_SIZE - 1)], bonus, CORR_MAX);
    gravity(nonPawnCorr[us][1][pos.nonPawnKey[1] & (CORR_SIZE - 1)], bonus, CORR_MAX);
    gravity(minorCorr[us][pos.minorKey & (CORR_SIZE - 1)], bonus, CORR_MAX);
    gravity(majorCorr[us][pos.majorKey & (CORR_SIZE - 1)], bonus, CORR_MAX);
    if ((ss - 1)->movedPiece != NO_PIECE && (ss - 2)->contCorr != &contCorr[12][0])
        gravity((*(ss - 2)->contCorr)[(ss - 1)->movedPiece][toSq((ss - 1)->move)], bonus, CORR_MAX);
}

int Searcher::quietHistory(const Position& pos, Stack* ss, Move m) const {
    const int pc = pos.board[fromSq(m)], to = toSq(m);
    return mainHist[pos.stm][(ss->threats >> fromSq(m)) & 1][(ss->threats >> to) & 1][m & 4095] + (*(ss - 1)->contHist)[pc][to] +
           (*(ss - 2)->contHist)[pc][to] + (*(ss - 4)->contHist)[pc][to] / 2 + pawnHist[pos.pawnKey & 1023][pc][to] / 2;
}

void Searcher::updateCont(Stack* ss, int pc, int to, int bonus) {
    for (int i : {1, 2, 4})
        if ((ss - i)->contHist != &contHist[12][0]) gravity((*(ss - i)->contHist)[pc][to], bonus);
}

void Searcher::updateQuiet(const Position& pos, Stack* ss, Move m, int bonus) {
    const int pc = pos.board[fromSq(m)], to = toSq(m);
    gravity(mainHist[pos.stm][(ss->threats >> fromSq(m)) & 1][(ss->threats >> to) & 1][m & 4095], bonus);
    gravity(pawnHist[pos.pawnKey & 1023][pc][to], bonus);
    updateCont(ss, pc, to, bonus);
}

// ---------------------------------------------------------------- move ordering

namespace {

enum { ST_TT, ST_GEN_NOISY, ST_GOOD_NOISY, ST_KILLER, ST_GEN_QUIET, ST_QUIET, ST_BAD_NOISY, ST_DONE };

struct MovePicker {
    const Position& pos;
    Searcher& s;
    Stack* ss;
    Move ttMove, killer;
    bool noisyOnly;   // qsearch / probcut: drop quiets and captures failing the SEE margin
    int seeMargin;
    bool skipQuiets = false;
    int stage = ST_TT;
    int n = 0, idx = 0, nBad = 0, badIdx = 0;
    Move moves[256];
    int scores[256];
    Move bad[256];

    MovePicker(const Position& p, Searcher& sr, Stack* st, Move tt, Move k, bool noisy, int margin)
        : pos(p), s(sr), ss(st), ttMove(tt), killer(k), noisyOnly(noisy), seeMargin(margin) {}

    Move pick() {
        int best = idx;
        for (int i = idx + 1; i < n; i++)
            if (scores[i] > scores[best]) best = i;
        std::swap(moves[idx], moves[best]);
        std::swap(scores[idx], scores[best]);
        return moves[idx++];
    }

    Move next() {
        switch (stage) {
        case ST_TT:
            stage = ST_GEN_NOISY;
            if (ttMove) return ttMove;
            [[fallthrough]];
        case ST_GEN_NOISY: {
            MoveList ml;
            generate(pos, ml, GEN_NOISY);
            n = ml.n;
            for (int i = 0; i < n; i++) {
                Move m = moves[i] = ml.moves[i];
                int sc = s.captEntry(pos, m) + 16 * SeeValue[pos.capturedPt(m)];
                if (isPromo(m)) sc += promoPt(m) == QUEEN ? 16000 : -40000;
                scores[i] = sc;
            }
            idx = 0;
            stage = ST_GOOD_NOISY;
        }
            [[fallthrough]];
        case ST_GOOD_NOISY:
            while (idx < n) {
                Move m = pick();
                if (m == ttMove) continue;
                if (!pos.see(m, noisyOnly ? seeMargin : -s.captEntry(pos, m) / 64)) {
                    if (!noisyOnly) bad[nBad++] = m;
                    continue;
                }
                return m;
            }
            if (noisyOnly) {
                stage = ST_DONE;
                return MOVE_NONE;
            }
            stage = ST_KILLER;
            [[fallthrough]];
        case ST_KILLER:
            stage = ST_GEN_QUIET;
            if (!skipQuiets && killer && killer != ttMove && !isNoisy(killer) && pos.isPseudoLegal(killer)) return killer;
            [[fallthrough]];
        case ST_GEN_QUIET:
            n = idx = 0;
            if (!skipQuiets) {
                MoveList ml;
                generate(pos, ml, GEN_QUIET);
                n = ml.n;
                for (int i = 0; i < n; i++) {
                    moves[i] = ml.moves[i];
                    scores[i] = s.quietHistory(pos, ss, ml.moves[i]);
                }
            }
            stage = ST_QUIET;
            [[fallthrough]];
        case ST_QUIET:
            if (!skipQuiets)
                while (idx < n) {
                    Move m = pick();
                    if (m == ttMove || m == killer) continue;
                    return m;
                }
            stage = ST_BAD_NOISY;
            [[fallthrough]];
        case ST_BAD_NOISY:
            if (badIdx < nBad) return bad[badIdx++];
            stage = ST_DONE;
            [[fallthrough]];
        default:
            return MOVE_NONE;
        }
    }
};

} // namespace

// ---------------------------------------------------------------- search

int Searcher::search(const Position& pos, Stack* ss, int alpha, int beta, int depth, bool cutNode) {
    const bool pvNode = beta - alpha > 1, root = ss->ply == 0;
    if (depth <= 0) return qsearch(pos, ss, alpha, beta);
    ss->pvLen = 0;
    if (shouldStop()) return 0;
    if (ss->ply + 1 > selDepth) selDepth = ss->ply + 1;
    if (!root) {
        if (isDraw(pos, ss->ply)) return 0;
        if (ss->ply >= MAX_PLY - 1) return pos.checkers ? 0 : evaluate(pos, ss);
        alpha = std::max(alpha, -SCORE_MATE + ss->ply);
        beta = std::min(beta, SCORE_MATE - ss->ply - 1);
        if (alpha >= beta) return alpha;
    }

    const bool inCheck = pos.checkers;
    const Move excluded = ss->excluded;
    const int us = pos.stm, ply = ss->ply;
    (ss + 1)->killer = MOVE_NONE;
    (ss + 1)->excluded = MOVE_NONE;
    (ss + 2)->cutoffCnt = 0;

    // Transposition table
    bool ttHit = false;
    TTEntry* tte = tt->probe(pos.key, ttHit);
    Move ttMove = MOVE_NONE;
    int ttScore = SCORE_NONE, ttEval = SCORE_NONE, ttDepth = -1, ttBound = BOUND_NONE;
    bool ttPv = pvNode;
    if (ttHit) {
        TTEntry e = *tte;
        ttMove = e.move;
        ttScore = scoreFromTT(e.score, ply);
        ttEval = e.eval;
        ttDepth = e.depth();
        ttBound = e.bound();
        ttPv |= e.pv();
        if (ttMove && !pos.isPseudoLegal(ttMove)) ttMove = MOVE_NONE;
    }
    if (excluded) ttPv = ss->ttPv;
    ss->ttPv = ttPv;

    if (!pvNode && !excluded && ttDepth >= depth && ttScore != SCORE_NONE && (ttBound & (ttScore >= beta ? BOUND_LOWER : BOUND_UPPER)) && pos.rule50 < 90)
        return ttScore;

    // Static evaluation
    int rawEval = SCORE_NONE, eval = SCORE_NONE;
    ss->threats = pos.threatsBy(us ^ 1);
    if (inCheck)
        ss->staticEval = SCORE_NONE;
    else if (excluded)
        eval = ss->staticEval;
    else {
        rawEval = ttEval != SCORE_NONE ? ttEval : evaluate(pos, ss);
        eval = ss->staticEval = std::clamp(rawEval + correction(pos, ss), -MATE_BOUND + 1, MATE_BOUND - 1);
        if (ttScore != SCORE_NONE && (ttBound & (ttScore > eval ? BOUND_LOWER : BOUND_UPPER)))
            eval = ttScore;
        else if (!ttHit)
            tt->store(tte, pos.key, MOVE_NONE, SCORE_NONE, rawEval, 0, BOUND_NONE, ttPv);
    }

    bool improving = false;
    if (!inCheck) {
        if ((ss - 2)->staticEval != SCORE_NONE)
            improving = ss->staticEval > (ss - 2)->staticEval;
        else if ((ss - 4)->staticEval != SCORE_NONE)
            improving = ss->staticEval > (ss - 4)->staticEval;
        else
            improving = true;
    }

    // Internal iterative reduction
    if (depth >= 4 && !ttMove && !excluded && (pvNode || cutNode)) depth--;

    if (!pvNode && !inCheck && !excluded) {
        // Reverse futility pruning
        if (depth <= 8 && eval - 75 * (depth - improving) >= beta && eval < MATE_BOUND && beta > -MATE_BOUND) return (eval + beta) / 2;

        // Razoring
        if (depth <= 3 && eval + 280 * depth < alpha) {
            int v = qsearch(pos, ss, alpha, alpha + 1);
            if (v <= alpha) return v;
        }

        // Null move pruning
        if (depth >= 3 && (ss - 1)->move != MOVE_NULL && eval >= beta && ss->staticEval >= beta - 20 * depth + 180 && pos.hasNonPawn(us) && ply >= nmpMinPly &&
            beta > -MATE_BOUND) {
            int R = 4 + depth / 3 + std::min((eval - beta) / 200, 3);
            Position next = pos;
            next.makeNull();
            tt->prefetch(next.key);
            ss->move = MOVE_NULL;
            ss->movedPiece = NO_PIECE;
            ss->contHist = &contHist[12][0];
            ss->contCorr = &contCorr[12][0];
            keys[keyBase + ply + 1] = next.key;
            if (NNUE::loaded) nn.pushNull(ply + 1);
            nodes.store(nodeCount() + 1, std::memory_order_relaxed);
            int v = -search(next, ss + 1, -beta, -beta + 1, depth - R, !cutNode);
            if (stop->load(std::memory_order_relaxed)) return 0;
            if (v >= beta) {
                if (v >= MATE_BOUND) v = beta;
                if (nmpMinPly || depth < 14) return v;
                nmpMinPly = ply + 3 * (depth - R) / 4;
                int v2 = search(pos, ss, beta - 1, beta, depth - R, false);
                nmpMinPly = 0;
                if (v2 >= beta) return v;
            }
        }

        // ProbCut
        const int pcBeta = beta + 200 - 60 * improving;
        if (depth >= 5 && std::abs(beta) < MATE_BOUND && !(ttDepth >= depth - 3 && ttScore != SCORE_NONE && ttScore < pcBeta)) {
            MovePicker mp(pos, *this, ss, ttMove && isNoisy(ttMove) && pos.see(ttMove, pcBeta - ss->staticEval) ? ttMove : MOVE_NONE, MOVE_NONE, true,
                          pcBeta - ss->staticEval);
            while (Move m = mp.next()) {
                if (!pos.isLegal(m)) continue;
                Position next = pos;
                next.makeMove(m);
                tt->prefetch(next.key);
                ss->move = m;
                ss->movedPiece = pos.board[fromSq(m)];
                ss->contHist = &contHist[ss->movedPiece][toSq(m)];
                ss->contCorr = &contCorr[ss->movedPiece][toSq(m)];
                keys[keyBase + ply + 1] = next.key;
                if (NNUE::loaded) nn.push(ply + 1, pos, m);
                nodes.store(nodeCount() + 1, std::memory_order_relaxed);
                int v = -qsearch(next, ss + 1, -pcBeta, -pcBeta + 1);
                if (v >= pcBeta) v = -search(next, ss + 1, -pcBeta, -pcBeta + 1, depth - 4, !cutNode);
                if (stop->load(std::memory_order_relaxed)) return 0;
                if (v >= pcBeta) {
                    tt->store(tte, pos.key, m, scoreToTT(v, ply), rawEval, depth - 3, BOUND_LOWER, ttPv);
                    return v;
                }
            }
        }
    }

    // Move loop
    int bestScore = -SCORE_INF, moveCount = 0;
    Move bestMove = MOVE_NONE;
    const int origAlpha = alpha;
    Move quiets[64], noisies[32];
    int nQuiets = 0, nNoisies = 0;
    MovePicker mp(pos, *this, ss, ttMove, ss->killer, false, 0);

    while (Move m = mp.next()) {
        if (m == excluded || !pos.isLegal(m)) continue;
        moveCount++;
        const bool quiet = !isNoisy(m);
        const int hist = quiet ? quietHistory(pos, ss, m) : captEntry(pos, m);

        // Shallow-depth pruning
        if (!root && bestScore > -MATE_BOUND && pos.hasNonPawn(us)) {
            int lmrDepth = std::max(0, depth - LmrTable[quiet][std::min(depth, 63)][std::min(moveCount, 63)] / 1024);
            if (quiet) {
                if (moveCount >= (3 + depth * depth) / (2 - improving)) mp.skipQuiets = true;
                if (!inCheck && lmrDepth <= 8 && ss->staticEval + 90 + 100 * lmrDepth <= alpha) {
                    mp.skipQuiets = true;
                    continue;
                }
                if (lmrDepth <= 4 && hist < -3500 * depth) continue;
                if (!pos.see(m, -25 * lmrDepth * lmrDepth)) continue;
            } else {
                if (!inCheck && lmrDepth <= 6 && ss->staticEval + 250 + 220 * lmrDepth + SeeValue[pos.capturedPt(m)] <= alpha) continue;
                if (!pos.see(m, -95 * depth)) continue;
            }
        }

        // Singular extension
        int ext = 0;
        if (!root && depth >= 7 && m == ttMove && !excluded && ttDepth >= depth - 3 && (ttBound & BOUND_LOWER) && std::abs(ttScore) < MATE_BOUND && ply < 2 * rootDepth) {
            const int sBeta = ttScore - depth, sDepth = (depth - 1) / 2;
            ss->excluded = m;
            int v = search(pos, ss, sBeta - 1, sBeta, sDepth, cutNode);
            ss->excluded = MOVE_NONE;
            if (stop->load(std::memory_order_relaxed)) return 0;
            if (v < sBeta) {
                ext = 1;
                if (!pvNode && v < sBeta - 20) ext = 2;
                if (!pvNode && quiet && v < sBeta - 100) ext = 3;
            } else if (sBeta >= beta)
                return sBeta;
            else if (ttScore >= beta)
                ext = -2;
            else if (cutNode)
                ext = -2;
        }

        int newDepth = depth - 1 + ext;
        Position next = pos;
        next.makeMove(m);
        tt->prefetch(next.key);
        ss->move = m;
        ss->movedPiece = pos.board[fromSq(m)];
        ss->contHist = &contHist[ss->movedPiece][toSq(m)];
        ss->contCorr = &contCorr[ss->movedPiece][toSq(m)];
        keys[keyBase + ply + 1] = next.key;
        if (NNUE::loaded) nn.push(ply + 1, pos, m);
        const u64 nodesBefore = nodeCount();
        nodes.store(nodesBefore + 1, std::memory_order_relaxed);

        int v;
        if (depth >= 2 && moveCount > 1 + root) {
            int r = LmrTable[quiet][std::min(depth, 63)][std::min(moveCount, 63)];
            if (ttPv) r -= 1024;
            if (pvNode) r -= 512;
            if (cutNode) r += 1536;
            if (!improving) r += 512;
            if (next.checkers) r -= 1024;
            if ((ss + 1)->cutoffCnt > 3) r += 1024;
            if (quiet) {
                if (ttMove && isNoisy(ttMove)) r += 1024;
                if (m == ss->killer) r -= 1024;
                r -= hist * 1024 / 9000;
            } else
                r -= hist * 1024 / 6000;
            int d = std::clamp(newDepth - r / 1024, 1, newDepth);
            ss->reduction = newDepth - d;
            v = -search(next, ss + 1, -alpha - 1, -alpha, d, true);
            ss->reduction = 0;
            if (v > alpha && d < newDepth) {
                newDepth += (v > bestScore + 40 + 2 * newDepth) - (v < bestScore + newDepth);
                if (newDepth > d) v = -search(next, ss + 1, -alpha - 1, -alpha, newDepth, !cutNode);
                if (quiet) updateCont(ss, ss->movedPiece, toSq(m), v <= alpha ? -1000 : v >= beta ? 1000 : 0);
            }
        } else if (!pvNode || moveCount > 1)
            v = -search(next, ss + 1, -alpha - 1, -alpha, newDepth, !cutNode);
        if (pvNode && (moveCount == 1 || v > alpha)) v = -search(next, ss + 1, -beta, -alpha, newDepth, false);

        if (stop->load(std::memory_order_relaxed)) return 0;
        if (root) rootNodes[m & 4095] += nodeCount() - nodesBefore;

        if (v > bestScore) {
            bestScore = v;
            if (v > alpha) {
                bestMove = m;
                if (pvNode) {
                    ss->pv[0] = m;
                    memcpy(ss->pv + 1, (ss + 1)->pv, sizeof(Move) * (ss + 1)->pvLen);
                    ss->pvLen = (ss + 1)->pvLen + 1;
                }
                if (v >= beta) {
                    ss->cutoffCnt++;
                    break;
                }
                alpha = v;
            }
        }
        if (m != bestMove) {
            if (quiet && nQuiets < 64) quiets[nQuiets++] = m;
            if (!quiet && nNoisies < 32) noisies[nNoisies++] = m;
        }
    }

    if (moveCount == 0) return excluded ? alpha : inCheck ? -SCORE_MATE + ply : 0;

    if (bestScore >= beta) {
        const int bonus = std::min(170 * depth - 90, 1900), malus = std::min(160 * depth - 80, 1700);
        if (!isNoisy(bestMove)) {
            ss->killer = bestMove;
            updateQuiet(pos, ss, bestMove, bonus);
            for (int i = 0; i < nQuiets; i++) updateQuiet(pos, ss, quiets[i], -malus);
        } else
            gravity(captEntry(pos, bestMove), bonus);
        for (int i = 0; i < nNoisies; i++) gravity(captEntry(pos, noisies[i]), -malus);
    }

    const int bound = bestScore >= beta ? BOUND_LOWER : (pvNode && alpha > origAlpha) ? BOUND_EXACT : BOUND_UPPER;
    if (!excluded) {
        tt->store(tte, pos.key, bestMove, scoreToTT(bestScore, ply), rawEval, depth, bound, ttPv);
        if (!inCheck && (!bestMove || !isNoisy(bestMove)) && !(bound == BOUND_LOWER && bestScore <= ss->staticEval) &&
            !(bound == BOUND_UPPER && bestScore >= ss->staticEval))
            updateCorrection(pos, ss, bestScore - ss->staticEval, depth);
    }
    return bestScore;
}

int Searcher::qsearch(const Position& pos, Stack* ss, int alpha, int beta) {
    const bool pvNode = beta - alpha > 1;
    const int ply = ss->ply;
    ss->pvLen = 0;
    if (shouldStop()) return 0;
    if (ply + 1 > selDepth) selDepth = ply + 1;
    if (isDraw(pos, ply)) return 0;
    if (ply >= MAX_PLY - 1) return pos.checkers ? 0 : evaluate(pos, ss);
    const bool inCheck = pos.checkers;

    bool ttHit = false;
    TTEntry* tte = tt->probe(pos.key, ttHit);
    Move ttMove = MOVE_NONE;
    int ttScore = SCORE_NONE, ttEval = SCORE_NONE, ttBound = BOUND_NONE;
    bool ttPv = pvNode;
    if (ttHit) {
        TTEntry e = *tte;
        ttMove = e.move;
        ttScore = scoreFromTT(e.score, ply);
        ttEval = e.eval;
        ttBound = e.bound();
        ttPv |= e.pv();
        if (ttMove && ((!inCheck && !isNoisy(ttMove)) || !pos.isPseudoLegal(ttMove))) ttMove = MOVE_NONE;
    }
    if (!pvNode && ttScore != SCORE_NONE && (ttBound & (ttScore >= beta ? BOUND_LOWER : BOUND_UPPER))) return ttScore;

    int rawEval = SCORE_NONE, bestScore = -SCORE_INF, futility = -SCORE_INF;
    if (inCheck) {
        ss->staticEval = SCORE_NONE;
        ss->threats = pos.threatsBy(pos.stm ^ 1);
    } else {
        rawEval = ttEval != SCORE_NONE ? ttEval : evaluate(pos, ss);
        bestScore = ss->staticEval = std::clamp(rawEval + correction(pos, ss), -MATE_BOUND + 1, MATE_BOUND - 1);
        if (ttScore != SCORE_NONE && (ttBound & (ttScore > bestScore ? BOUND_LOWER : BOUND_UPPER))) bestScore = ttScore;
        if (bestScore >= beta) {
            if (!ttHit) tt->store(tte, pos.key, MOVE_NONE, scoreToTT(bestScore, ply), rawEval, 0, BOUND_LOWER, ttPv);
            return bestScore;
        }
        if (bestScore > alpha) alpha = bestScore;
        futility = ss->staticEval + 130;
    }

    Move bestMove = MOVE_NONE;
    int moveCount = 0;
    MovePicker mp(pos, *this, ss, ttMove, MOVE_NONE, !inCheck, 0);
    while (Move m = mp.next()) {
        if (!pos.isLegal(m)) continue;
        moveCount++;
        if (bestScore > -MATE_BOUND) {
            if (inCheck) {
                mp.skipQuiets = true;
                if (!isNoisy(m)) continue;
            } else if (!isPromo(m) && futility + SeeValue[pos.capturedPt(m)] <= alpha) {
                bestScore = std::max(bestScore, futility + SeeValue[pos.capturedPt(m)]);
                continue;
            }
        }
        Position next = pos;
        next.makeMove(m);
        tt->prefetch(next.key);
        ss->move = m;
        ss->movedPiece = pos.board[fromSq(m)];
        ss->contHist = &contHist[ss->movedPiece][toSq(m)];
        ss->contCorr = &contCorr[ss->movedPiece][toSq(m)];
        keys[keyBase + ply + 1] = next.key;
        if (NNUE::loaded) nn.push(ply + 1, pos, m);
        nodes.store(nodeCount() + 1, std::memory_order_relaxed);
        int v = -qsearch(next, ss + 1, -beta, -alpha);
        if (stop->load(std::memory_order_relaxed)) return 0;
        if (v > bestScore) {
            bestScore = v;
            if (v > alpha) {
                bestMove = m;
                if (pvNode) {
                    ss->pv[0] = m;
                    memcpy(ss->pv + 1, (ss + 1)->pv, sizeof(Move) * (ss + 1)->pvLen);
                    ss->pvLen = (ss + 1)->pvLen + 1;
                }
                if (v >= beta) break;
                alpha = v;
            }
        }
    }
    if (inCheck && moveCount == 0) return -SCORE_MATE + ply;
    tt->store(tte, pos.key, bestMove, scoreToTT(bestScore, ply), rawEval, 0, bestScore >= beta ? BOUND_LOWER : BOUND_UPPER, ttPv);
    return bestScore;
}

// ---------------------------------------------------------------- iterative deepening

void Searcher::think() {
    for (int i = 0; i < MAX_PLY + 12; i++) {
        Stack& s = stack[i];
        s.move = s.killer = s.excluded = MOVE_NONE;
        s.ply = i - 6;
        s.staticEval = SCORE_NONE;
        s.movedPiece = NO_PIECE;
        s.cutoffCnt = s.reduction = 0;
        s.ttPv = false;
        s.threats = 0;
        s.contHist = &contHist[12][0];
        s.contCorr = &contCorr[12][0];
        s.pvLen = 0;
    }
    Stack* ss = stack + 6;
    memset(rootNodes, 0, sizeof(rootNodes));
    nodes.store(0, std::memory_order_relaxed);
    nmpMinPly = 0;
    bestMove = MOVE_NONE;
    bestScore = 0;
    completedDepth = 0;
    rootPvLen = 0;
    if (NNUE::loaded) nn.reset();

    int stability = 0;
    for (rootDepth = 1; rootDepth <= lim.depth; rootDepth++) {
        selDepth = 0;
        int alpha = -SCORE_INF, beta = SCORE_INF, delta = 12, d = rootDepth, score = 0;
        if (rootDepth >= 4) {
            alpha = std::max(-SCORE_INF, bestScore - delta);
            beta = std::min(SCORE_INF, bestScore + delta);
        }
        for (;;) {
            score = search(rootPos, ss, alpha, beta, std::max(1, d), false);
            if (stop->load(std::memory_order_relaxed)) break;
            if (score <= alpha) {
                beta = (alpha + beta) / 2;
                alpha = std::max(-SCORE_INF, score - delta);
                d = rootDepth;
            } else if (score >= beta) {
                beta = std::min(SCORE_INF, score + delta);
                if (d > rootDepth - 3 && std::abs(score) < MATE_BOUND) d--;
            } else
                break;
            delta += delta / 2;
            if (delta > 1200) alpha = -SCORE_INF, beta = SCORE_INF;
        }
        if (stop->load(std::memory_order_relaxed)) break;

        stability = ss->pv[0] == bestMove ? stability + 1 : 0;
        bestMove = ss->pv[0];
        bestScore = score;
        completedDepth = rootDepth;
        rootPvLen = ss->pvLen;
        memcpy(rootPv, ss->pv, sizeof(Move) * ss->pvLen);

        if (!isMain) continue;
        const i64 elapsed = nowMs() - lim.start;
        const u64 n = nodeCount();
        if (!silent) {
            printf("info depth %d seldepth %d score ", rootDepth, selDepth);
            if (std::abs(score) >= MATE_BOUND)
                printf("mate %d", score > 0 ? (SCORE_MATE - score + 1) / 2 : -(SCORE_MATE + score) / 2);
            else
                printf("cp %d", score);
            u64 total = 0;
            if (pool)
                for (auto& t : *pool) total += t->nodeCount();
            else
                total = n;
            printf(" nodes %llu nps %llu time %lld hashfull %d pv", (unsigned long long)total, (unsigned long long)(total * 1000 / std::max<i64>(elapsed, 1)),
                   (long long)elapsed, tt->hashfull());
            for (int i = 0; i < rootPvLen; i++) printf(" %s", moveToStr(rootPv[i]).c_str());
            printf("\n");
            fflush(stdout);
        }
        if (lim.softNodes && n >= lim.softNodes) break;
        if (lim.softMs >= 0) {
            double scale = 1.0;
            if (rootDepth >= 6) {
                double frac = double(rootNodes[bestMove & 4095]) / double(std::max<u64>(n, 1));
                scale = (1.55 - frac) * 1.35 * std::max(0.75, 1.5 - 0.1 * stability);
            }
            if (elapsed >= std::min<double>(lim.softMs * scale, lim.hardMs)) break;
        }
    }
    if (bestMove == MOVE_NONE) {
        MoveList ml;
        generateLegal(rootPos, ml);
        if (ml.n) bestMove = ml.moves[0];
    }
}
