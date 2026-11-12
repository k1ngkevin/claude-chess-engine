#pragma once
#include "types.h"
#include <cstdlib>
#include <cstring>

enum { BOUND_NONE = 0, BOUND_UPPER = 1, BOUND_LOWER = 2, BOUND_EXACT = 3 };

struct TTEntry {
    u16 key;
    Move move;
    i16 score;
    i16 eval;
    u8 depth8;   // depth + 1; 0 means empty
    u8 genBound; // generation (5 bits) | pv (1 bit) | bound (2 bits)
    int depth() const { return depth8 - 1; }
    int bound() const { return genBound & 3; }
    bool pv() const { return genBound & 4; }
};

struct TTBucket {
    TTEntry e[3];
    u16 pad;
};
static_assert(sizeof(TTBucket) == 32);

class TT {
    TTBucket* table = nullptr;
    u64 count = 0;
    u8 gen = 0;

public:
    ~TT() { free(table); }
    void resize(size_t mb) {
        free(table);
        count = mb * 1024 * 1024 / sizeof(TTBucket);
        table = (TTBucket*)aligned_alloc(64, count * sizeof(TTBucket));
        clear();
    }
    void clear() {
        memset((void*)table, 0, count * sizeof(TTBucket));
        gen = 0;
    }
    void newSearch() { gen += 8; }
    TTBucket* bucket(Key k) const { return &table[(u64)(((unsigned __int128)k * count) >> 64)]; }
    void prefetch(Key k) const { __builtin_prefetch(bucket(k)); }

    TTEntry* probe(Key k, bool& hit) const {
        TTEntry* e = bucket(k)->e;
        u16 k16 = (u16)k;
        for (int i = 0; i < 3; i++)
            if (e[i].key == k16 && e[i].depth8) {
                e[i].genBound = u8(gen | (e[i].genBound & 7));
                hit = true;
                return &e[i];
            }
        TTEntry* r = e;
        for (int i = 1; i < 3; i++)
            if (e[i].depth8 - ((gen - e[i].genBound) & 0xF8) < r->depth8 - ((gen - r->genBound) & 0xF8)) r = &e[i];
        hit = false;
        return r;
    }

    void store(TTEntry* e, Key k, Move m, int score, int eval, int depth, int bound, bool pv) const {
        u16 k16 = (u16)k;
        if (m || k16 != e->key) e->move = m;
        if (bound == BOUND_EXACT || k16 != e->key || depth + 1 + 2 * pv > e->depth8 - 4 || (e->genBound & 0xF8) != gen) {
            e->key = k16;
            e->score = (i16)score;
            e->eval = (i16)eval;
            e->depth8 = u8(depth + 1);
            e->genBound = u8(gen | (pv << 2) | bound);
        }
    }

    int hashfull() const {
        int n = 0;
        for (int i = 0; i < 1000; i++)
            for (auto& e : table[i].e) n += e.depth8 && (e.genBound & 0xF8) == gen;
        return n / 3;
    }
};
