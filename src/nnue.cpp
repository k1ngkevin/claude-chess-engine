#include "nnue.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#ifdef EVALFILE
// Embed the network file into the binary.
#if defined(__APPLE__)
__asm__(".section __DATA,__const\n.globl _gEmbeddedNet\n.p2align 6\n_gEmbeddedNet:\n.incbin \"" EVALFILE "\"\n.globl _gEmbeddedNetEnd\n_gEmbeddedNetEnd:\n.byte 0\n.text\n");
#else
__asm__(".section .rodata\n.globl gEmbeddedNet\n.p2align 6\ngEmbeddedNet:\n.incbin \"" EVALFILE "\"\n.globl gEmbeddedNetEnd\ngEmbeddedNetEnd:\n.byte 0\n.text\n");
#endif
extern "C" const unsigned char gEmbeddedNet[], gEmbeddedNetEnd[];
#endif

namespace NNUE {

bool loaded = false;
static Net* net = nullptr;

static bool loadBytes(const unsigned char* data, size_t size) {
    NetHeader h;
    const size_t body = sizeof(i16) * (size_t(INPUTS) * HIDDEN + HIDDEN + OUTPUT_BUCKETS * 2 * HIDDEN + OUTPUT_BUCKETS);
    if (size < sizeof(h) + body) return false;
    memcpy(&h, data, sizeof(h));
    if (h.magic != NET_MAGIC || h.hidden != HIDDEN || h.layout != NNUE_LAYOUT || h.outputBuckets != OUTPUT_BUCKETS) return false;
    if (!net) net = (Net*)aligned_alloc(64, (sizeof(Net) + 63) / 64 * 64);
    const unsigned char* p = data + sizeof(h);
    memcpy(net->ftW, p, sizeof(i16) * size_t(INPUTS) * HIDDEN);
    p += sizeof(i16) * size_t(INPUTS) * HIDDEN;
    memcpy(net->ftB, p, sizeof(i16) * HIDDEN);
    p += sizeof(i16) * HIDDEN;
    memcpy(net->outW, p, sizeof(i16) * OUTPUT_BUCKETS * 2 * HIDDEN);
    p += sizeof(i16) * OUTPUT_BUCKETS * 2 * HIDDEN;
    memcpy(net->outB, p, sizeof(i16) * OUTPUT_BUCKETS);
    loaded = true;
    return true;
}

bool loadFile(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char* buf = (unsigned char*)malloc(size);
    bool ok = fread(buf, 1, size, f) == (size_t)size && loadBytes(buf, size);
    free(buf);
    fclose(f);
    return ok;
}

bool loadEmbedded() {
#ifdef EVALFILE
    return loadBytes(gEmbeddedNet, size_t(gEmbeddedNetEnd - gEmbeddedNet));
#else
    return false;
#endif
}

static inline void addSub(i16* dst, const i16* src, const i16* a, const i16* s) {
    for (int i = 0; i < HIDDEN; i++) dst[i] = i16(src[i] + a[i] - s[i]);
}
static inline void addSubSub(i16* dst, const i16* src, const i16* a, const i16* s1, const i16* s2) {
    for (int i = 0; i < HIDDEN; i++) dst[i] = i16(src[i] + a[i] - s1[i] - s2[i]);
}
static inline void addAddSubSub(i16* dst, const i16* src, const i16* a1, const i16* a2, const i16* s1, const i16* s2) {
    for (int i = 0; i < HIDDEN; i++) dst[i] = i16(src[i] + a1[i] + a2[i] - s1[i] - s2[i]);
}

static inline bool kingCrosses(int persp, int from, int to) {
    if (NNUE_LAYOUT == 0) return false;
    int f = from ^ (persp ? 56 : 0), t = to ^ (persp ? 56 : 0);
    return ((f & 7) > 3) != ((t & 7) > 3) || LAYOUT.map[f] != LAYOUT.map[t];
}

void State::reset() {
    for (auto& p : finny)
        for (auto& m : p)
            for (auto& e : m) {
                memcpy(e.v, net->ftB, sizeof(e.v));
                memset(e.bb, 0, sizeof(e.bb));
            }
    acc[0].computed[0] = acc[0].computed[1] = false;
    acc[0].needsRefresh[0] = acc[0].needsRefresh[1] = true;
}

void State::push(int ply, const Position& pos, Move m) {
    Accumulator& a = acc[ply];
    a.computed[0] = a.computed[1] = false;
    a.needsRefresh[0] = a.needsRefresh[1] = false;
    const int from = fromSq(m), to = toSq(m), flag = flagOf(m), pc = pos.board[from], us = pos.stm;
    a.nAdd = a.nSub = 1;
    a.sub[0] = pc << 6 | from;
    a.add[0] = (isPromo(m) ? makePiece(us, promoPt(m)) : pc) << 6 | to;
    if (flag == EP)
        a.sub[a.nSub++] = makePiece(us ^ 1, PAWN) << 6 | (us == WHITE ? to - 8 : to + 8);
    else if (isCapture(m))
        a.sub[a.nSub++] = pos.board[to] << 6 | to;
    else if (flag == KING_CASTLE) {
        a.sub[a.nSub++] = makePiece(us, ROOK) << 6 | (to + 1);
        a.add[a.nAdd++] = makePiece(us, ROOK) << 6 | (to - 1);
    } else if (flag == QUEEN_CASTLE) {
        a.sub[a.nSub++] = makePiece(us, ROOK) << 6 | (to - 2);
        a.add[a.nAdd++] = makePiece(us, ROOK) << 6 | (to + 1);
    }
    if (pc % 6 == KING) a.needsRefresh[us] = kingCrosses(us, from, to);
}

void State::pushNull(int ply) {
    Accumulator& a = acc[ply];
    a.computed[0] = a.computed[1] = false;
    a.needsRefresh[0] = a.needsRefresh[1] = false;
    a.nAdd = a.nSub = 0;
}

void State::refresh(const Position& pos, int ply, int persp) {
    const int ksq = pos.kingSq(persp);
    const int k = ksq ^ (persp ? 56 : 0);
    FinnyEntry& e = finny[persp][LAYOUT.mirror && (k & 7) > 3][LAYOUT.map[k]];
    for (int pc = 0; pc < 12; pc++) {
        Bitboard cur = pos.pieces[pc % 6] & pos.colors[pc >= 6];
        Bitboard adds = cur & ~e.bb[pc], subs = e.bb[pc] & ~cur;
        while (adds) {
            const i16* w = &net->ftW[featureIndex(LAYOUT, persp, ksq, pc, poplsb(adds)) * HIDDEN];
            for (int i = 0; i < HIDDEN; i++) e.v[i] += w[i];
        }
        while (subs) {
            const i16* w = &net->ftW[featureIndex(LAYOUT, persp, ksq, pc, poplsb(subs)) * HIDDEN];
            for (int i = 0; i < HIDDEN; i++) e.v[i] -= w[i];
        }
        e.bb[pc] = cur;
    }
    memcpy(acc[ply].v[persp], e.v, sizeof(e.v));
    acc[ply].computed[persp] = true;
}

void State::update(const Position& pos, int ply, int persp) {
    // Walk back to the nearest usable accumulator for this perspective.
    int i = ply;
    while (!acc[i].computed[persp]) {
        if (acc[i].needsRefresh[persp]) {
            refresh(pos, ply, persp);
            return;
        }
        i--;
    }
    const int ksq = pos.kingSq(persp);
    auto W = [&](int code) { return &net->ftW[featureIndex(LAYOUT, persp, ksq, code >> 6, code & 63) * HIDDEN]; };
    for (int j = i + 1; j <= ply; j++) {
        Accumulator& a = acc[j];
        const i16* src = acc[j - 1].v[persp];
        i16* dst = a.v[persp];
        if (a.nAdd == 0)
            memcpy(dst, src, sizeof(i16) * HIDDEN);
        else if (a.nAdd == 1 && a.nSub == 1)
            addSub(dst, src, W(a.add[0]), W(a.sub[0]));
        else if (a.nAdd == 1)
            addSubSub(dst, src, W(a.add[0]), W(a.sub[0]), W(a.sub[1]));
        else
            addAddSubSub(dst, src, W(a.add[0]), W(a.add[1]), W(a.sub[0]), W(a.sub[1]));
        a.computed[persp] = true;
    }
}

static inline i64 screluDot(const i16* a, const i16* w) {
#if defined(__ARM_NEON)
    int32x4_t s0 = vdupq_n_s32(0), s1 = vdupq_n_s32(0);
    const int16x8_t zero = vdupq_n_s16(0), qa = vdupq_n_s16(QA);
    for (int i = 0; i < HIDDEN; i += 8) {
        int16x8_t v = vminq_s16(vmaxq_s16(vld1q_s16(a + i), zero), qa);
        int16x8_t wv = vld1q_s16(w + i);
        int32x4_t lo = vmull_s16(vget_low_s16(v), vget_low_s16(wv)), hi = vmull_high_s16(v, wv);
        s0 = vmlaq_s32(s0, lo, vmovl_s16(vget_low_s16(v)));
        s1 = vmlaq_s32(s1, hi, vmovl_high_s16(v));
    }
    return (i64)vaddlvq_s32(s0) + vaddlvq_s32(s1);
#else
    i64 sum = 0;
    for (int i = 0; i < HIDDEN; i++) {
        i32 v = a[i] < 0 ? 0 : a[i] > QA ? QA : a[i];
        sum += v * w[i] * v;
    }
    return sum;
#endif
}

int State::evaluate(const Position& pos, int ply) {
    if (!acc[ply].computed[0]) update(pos, ply, 0);
    if (!acc[ply].computed[1]) update(pos, ply, 1);
    const int b = outputBucket(popcount(pos.occ()));
    const int us = pos.stm;
    i64 sum = screluDot(acc[ply].v[us], net->outW[b]) + screluDot(acc[ply].v[us ^ 1], net->outW[b] + HIDDEN);
    return int((sum / QA + net->outB[b]) * SCALE / (QA * QB));
}

} // namespace NNUE
