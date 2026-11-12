#pragma once
// Network architecture shared by the engine and the trainer.
//   (768 * kingBuckets -> HIDDEN) x 2 perspectives -> SCReLU -> OUTPUT_BUCKETS
#include <cstdint>

namespace NNUE {

constexpr int QA = 255;    // accumulator quantisation
constexpr int QB = 256;    // output weight quantisation
constexpr int SCALE = 400; // network output -> centipawns
constexpr int OUTPUT_BUCKETS = 8;
constexpr uint32_t NET_MAGIC = 0x314E4E48; // "HNN1"

struct Layout {
    bool mirror; // flip horizontally so the king is always on files a-d
    int buckets;
    int map[64]; // king square (from own perspective, after mirroring) -> bucket
};

constexpr Layout makeLayout(int id) {
    Layout l{};
    l.mirror = id >= 1;
    l.buckets = id == 2 ? 4 : id == 3 ? 8 : 1;
    for (int s = 0; s < 64; s++) {
        int r = s >> 3, f = s & 7;
        if (f > 3) f = 7 - f;
        int b = 0;
        if (id == 2) b = r == 0 ? (f < 2 ? 0 : 1) : r == 1 ? 2 : 3;
        if (id == 3) b = r == 0 ? f : r == 1 ? 4 + f / 2 : r <= 3 ? 6 : 7;
        l.map[s] = b;
    }
    return l;
}

struct NetHeader {
    uint32_t magic, hidden, layout, outputBuckets;
};

// Feature index for piece `pc` (colour * 6 + type) on `sq`, seen by `persp` whose king is on `ksq`.
constexpr int featureIndex(const Layout& l, int persp, int ksq, int pc, int sq) {
    int flip = persp ? 56 : 0;
    int k = ksq ^ flip;
    if (l.mirror && (k & 7) > 3) flip ^= 7;
    int c = pc >= 6, pt = pc % 6;
    return l.map[k] * 768 + (c != persp) * 384 + pt * 64 + (sq ^ flip);
}

inline int outputBucket(int pieceCount) { return (pieceCount - 2) / 4; }

} // namespace NNUE
