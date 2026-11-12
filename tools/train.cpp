// NNUE trainer: multi-threaded CPU minibatch AdamW on self-play data.
//
//   train -data a.bin,b.bin -out net.bin [-hidden 256] [-layout 0] [-epochs 30] [-lr 0.001]
//         [-wdl 0.4] [-batch 16384] [-threads 10] [-lrfinal 0.02] [-seed 1]
//
// Model: (768 * kingBuckets -> hidden) x 2 -> SCReLU -> 1 (x 8 output buckets), trained to
// predict sigmoid(score / 400) blended with the game result.
#include "../src/datagen.h"
#include "../src/nnue_arch.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <functional>
#include <mutex>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace NNUE;

// ---------------------------------------------------------------- thread pool
class Pool {
    std::vector<std::thread> threads;
    std::mutex m;
    std::condition_variable cvStart, cvDone;
    std::function<void(int)> job;
    int generation = 0, running = 0;
    bool quit = false;

public:
    explicit Pool(int n) {
        for (int i = 0; i < n; i++)
            threads.emplace_back([this, i] {
                int seen = 0;
                for (;;) {
                    std::unique_lock<std::mutex> lk(m);
                    cvStart.wait(lk, [&] { return quit || generation != seen; });
                    if (quit) return;
                    seen = generation;
                    lk.unlock();
                    job(i);
                    lk.lock();
                    if (--running == 0) cvDone.notify_one();
                }
            });
    }
    ~Pool() {
        {
            std::lock_guard<std::mutex> lk(m);
            quit = true;
        }
        cvStart.notify_all();
        for (auto& t : threads) t.join();
    }
    int size() const { return (int)threads.size(); }
    void run(std::function<void(int)> f) {
        std::unique_lock<std::mutex> lk(m);
        job = std::move(f);
        running = size();
        generation++;
        cvStart.notify_all();
        cvDone.wait(lk, [&] { return running == 0; });
    }
};

// ---------------------------------------------------------------- data
struct Dataset {
    std::vector<std::pair<const PackedPos*, size_t>> chunks;
    std::vector<size_t> starts;
    size_t total = 0;
    void add(const std::string& path) {
        int fd = open(path.c_str(), O_RDONLY);
        struct stat st;
        if (fd < 0 || fstat(fd, &st)) {
            fprintf(stderr, "cannot open %s\n", path.c_str());
            exit(1);
        }
        size_t n = st.st_size / sizeof(PackedPos);
        void* p = mmap(nullptr, n * sizeof(PackedPos), PROT_READ, MAP_PRIVATE, fd, 0);
        if (p == MAP_FAILED) {
            fprintf(stderr, "mmap failed for %s\n", path.c_str());
            exit(1);
        }
        starts.push_back(total);
        chunks.push_back({(const PackedPos*)p, n});
        total += n;
        printf("loaded %s: %zu positions\n", path.c_str(), n);
    }
    const PackedPos& at(size_t i) const {
        size_t c = std::upper_bound(starts.begin(), starts.end(), i) - starts.begin() - 1;
        return chunks[c].first[i - starts[c]];
    }
};

// ---------------------------------------------------------------- model
static int H = 256, layoutId = 0, inputs = 768;
static Layout layout;
static size_t oW1, oB1, oW2, oB2, nParams;
static float wdlWeight = 0.4f;

static float* allocF(size_t n) {
    void* p = aligned_alloc(64, (n * sizeof(float) + 63) / 64 * 64);
    memset(p, 0, n * sizeof(float));
    return (float*)p;
}

struct Scratch {
    float *acc[2], *act[2], *d[2];
    double loss = 0;
    Scratch() {
        for (int i = 0; i < 2; i++) acc[i] = allocF(H), act[i] = allocF(H), d[i] = allocF(H);
    }
};

static thread_local float lastOut = 0;
static inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// Forward (and backward if G != nullptr) for one position. Returns squared error.
static float process(const PackedPos& p, const float* P, float* G, Scratch& sc) {
    int pcs[32], sqs[32], n = 0, ksq[2] = {0, 0};
    for (u64 occ = p.occ; occ; occ &= occ - 1, n++) {
        int sq = __builtin_ctzll(occ);
        int pc = (p.pcs[n / 2] >> (4 * (n & 1))) & 15;
        pcs[n] = pc;
        sqs[n] = sq;
        if (pc == 5) ksq[0] = sq;
        if (pc == 11) ksq[1] = sq;
    }
    const int us = p.stm;
    int feats[2][32];
    for (int i = 0; i < n; i++) {
        feats[0][i] = featureIndex(layout, us, ksq[us], pcs[i], sqs[i]);
        feats[1][i] = featureIndex(layout, us ^ 1, ksq[us ^ 1], pcs[i], sqs[i]);
    }
    const int bucket = outputBucket(n);
    const float* w2 = P + oW2 + (size_t)bucket * 2 * H;
    float out = P[oB2 + bucket];
    for (int s = 0; s < 2; s++) {
        float* acc = sc.acc[s];
        memcpy(acc, P + oB1, sizeof(float) * H);
        for (int i = 0; i < n; i++) {
            const float* w = P + oW1 + (size_t)feats[s][i] * H;
            for (int j = 0; j < H; j++) acc[j] += w[j];
        }
        float* act = sc.act[s];
        const float* w = w2 + s * H;
        float sum = 0;
        for (int j = 0; j < H; j++) {
            float a = std::min(std::max(acc[j], 0.0f), 1.0f);
            act[j] = a;
            sum += a * a * w[j];
        }
        out += sum;
    }
    lastOut = out;
    const float pred = sigmoid(out);
    const float cp = us == WHITE ? p.score : -p.score;
    const float wdl = us == WHITE ? p.result / 2.0f : 1.0f - p.result / 2.0f;
    const float target = wdlWeight * wdl + (1 - wdlWeight) * sigmoid(cp / SCALE);
    const float err = pred - target;
    if (!G) return err * err;

    const float g = 2 * err * pred * (1 - pred);
    G[oB2 + bucket] += g;
    float* gw2 = G + oW2 + (size_t)bucket * 2 * H;
    float* gb1 = G + oB1;
    for (int s = 0; s < 2; s++) {
        const float *act = sc.act[s], *w = w2 + s * H;
        float *d = sc.d[s], *gw = gw2 + s * H;
        for (int j = 0; j < H; j++) {
            float a = act[j];
            gw[j] += g * a * a;
            float dj = (a > 0.0f && a < 1.0f) ? g * w[j] * 2 * a : 0.0f;
            d[j] = dj;
            gb1[j] += dj;
        }
        for (int i = 0; i < n; i++) {
            float* gw1 = G + oW1 + (size_t)feats[s][i] * H;
            for (int j = 0; j < H; j++) gw1[j] += d[j];
        }
    }
    return err * err;
}

static void save(const std::string& path, const float* P) {
    FILE* f = fopen(path.c_str(), "wb");
    NetHeader h{NET_MAGIC, (uint32_t)H, (uint32_t)layoutId, OUTPUT_BUCKETS};
    fwrite(&h, sizeof(h), 1, f);
    auto dump = [&](size_t off, size_t n, float scale) {
        std::vector<i16> q(n);
        for (size_t i = 0; i < n; i++) q[i] = (i16)std::lround(std::min(std::max(P[off + i] * scale, -32767.0f), 32767.0f));
        fwrite(q.data(), sizeof(i16), n, f);
    };
    dump(oW1, (size_t)inputs * H, QA);
    dump(oB1, H, QA);
    dump(oW2, (size_t)OUTPUT_BUCKETS * 2 * H, QB);
    dump(oB2, OUTPUT_BUCKETS, QA * QB);
    fclose(f);
}

int main(int argc, char** argv) {
    std::string dataArg, out = "net.bin";
    int epochs = 30, batch = 16384, nThreads = 10;
    float lr = 0.001f, lrFinal = 0.02f;
    u64 seed = 1;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string a = argv[i], v = argv[i + 1];
        if (a == "-data") dataArg = v;
        else if (a == "-out") out = v;
        else if (a == "-hidden") H = atoi(v.c_str());
        else if (a == "-layout") layoutId = atoi(v.c_str());
        else if (a == "-epochs") epochs = atoi(v.c_str());
        else if (a == "-lr") lr = (float)atof(v.c_str());
        else if (a == "-lrfinal") lrFinal = (float)atof(v.c_str());
        else if (a == "-wdl") wdlWeight = (float)atof(v.c_str());
        else if (a == "-batch") batch = atoi(v.c_str());
        else if (a == "-threads") nThreads = atoi(v.c_str());
        else if (a == "-seed") seed = strtoull(v.c_str(), nullptr, 10);
    }
    Dataset ds;
    for (size_t pos = 0; pos < dataArg.size();) {
        size_t c = dataArg.find(',', pos);
        if (c == std::string::npos) c = dataArg.size();
        ds.add(dataArg.substr(pos, c - pos));
        pos = c + 1;
    }
    if (!ds.total) {
        fprintf(stderr, "usage: train -data file[,file] -out net.bin [-hidden N] [-layout N] [-epochs N] [-lr F] [-wdl F]\n");
        return 1;
    }
    layout = makeLayout(layoutId);
    inputs = layout.buckets * 768;
    oW1 = 0;
    oB1 = oW1 + (size_t)inputs * H;
    oW2 = oB1 + H;
    oB2 = oW2 + (size_t)OUTPUT_BUCKETS * 2 * H;
    nParams = oB2 + OUTPUT_BUCKETS;

    float *P = allocF(nParams), *M = allocF(nParams), *V = allocF(nParams);
    u64 rs = seed * 0x9E3779B97F4A7C15ULL + 1;
    auto rnd = [&]() {
        rs ^= rs >> 12, rs ^= rs << 25, rs ^= rs >> 27;
        return rs * 0x2545F4914F6CDD1DULL;
    };
    auto uni = [&](float r) { return ((rnd() >> 40) / float(1 << 24) * 2 - 1) * r; };
    for (size_t i = oW1; i < oB1; i++) P[i] = uni(0.1f);
    for (size_t i = oW2; i < oB2; i++) P[i] = uni(1.0f / std::sqrt(2.0f * H));

    Pool pool(nThreads);
    std::vector<float*> G(nThreads);
    std::vector<Scratch> scratch(nThreads);
    for (auto& g : G) g = allocF(nParams);

    const size_t nVal = std::min<size_t>(ds.total / 50, 1000000), nTrain = ds.total - nVal;
    std::vector<u32> order(nTrain);
    for (size_t i = 0; i < nTrain; i++) order[i] = (u32)i;
    const size_t batchesPerEpoch = nTrain / batch;
    const double totalSteps = (double)batchesPerEpoch * epochs;
    printf("params %zu, train %zu, val %zu, hidden %d, layout %d, %zu batches/epoch, wdl %.2f\n", nParams, nTrain, nVal, H, layoutId, batchesPerEpoch, wdlWeight);

    const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f, decay = 0.01f;
    double step = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (int epoch = 1; epoch <= epochs; epoch++) {
        for (size_t i = nTrain - 1; i > 0; i--) std::swap(order[i], order[rnd() % (i + 1)]);
        double epochLoss = 0;
        float curLr = lr;
        for (size_t b = 0; b < batchesPerEpoch; b++) {
            const u32* idx = order.data() + b * batch;
            pool.run([&](int t) {
                Scratch& sc = scratch[t];
                sc.loss = 0;
                size_t lo = (size_t)batch * t / nThreads, hi = (size_t)batch * (t + 1) / nThreads;
                for (size_t i = lo; i < hi; i++) sc.loss += process(ds.at(idx[i]), P, G[t], sc);
            });
            for (auto& sc : scratch) epochLoss += sc.loss;
            step += 1;
            curLr = lr * std::pow(lrFinal, (float)(step / totalSteps));
            const float corr = std::sqrt(1 - std::pow(b2, (float)step)) / (1 - std::pow(b1, (float)step));
            const float stepLr = curLr * corr, inv = 1.0f / batch, shrink = 1 - curLr * decay;
            pool.run([&](int t) {
                size_t lo = nParams * t / nThreads, hi = nParams * (t + 1) / nThreads;
                for (int k = 0; k < nThreads; k++) {
                    float* g = G[k];
                    if (k == 0) continue;
                    float* g0 = G[0];
                    for (size_t i = lo; i < hi; i++) g0[i] += g[i], g[i] = 0;
                }
                float* g0 = G[0];
                for (size_t i = lo; i < hi; i++) {
                    float g = g0[i] * inv;
                    g0[i] = 0;
                    M[i] = b1 * M[i] + (1 - b1) * g;
                    V[i] = b2 * V[i] + (1 - b2) * g * g;
                    float p = P[i] * shrink - stepLr * M[i] / (std::sqrt(V[i]) + eps);
                    P[i] = std::min(std::max(p, -1.98f), 1.98f);
                }
            });
        }
        // validation
        pool.run([&](int t) {
            Scratch& sc = scratch[t];
            sc.loss = 0;
            size_t lo = nVal * t / nThreads, hi = nVal * (t + 1) / nThreads;
            for (size_t i = lo; i < hi; i++) sc.loss += process(ds.at(nTrain + i), P, nullptr, sc);
        });
        double valLoss = 0;
        for (auto& sc : scratch) valLoss += sc.loss;
        double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        printf("epoch %d/%d  train %.6f  val %.6f  lr %.6f  %.0fs  %.0f pos/s\n", epoch, epochs, epochLoss / (batchesPerEpoch * (double)batch), valLoss / std::max<size_t>(nVal, 1),
               curLr, secs, step * batch / secs);
        fflush(stdout);
        save(out, P);
    }
    // Reference outputs (centipawns, side to move) for the first records, to verify engine inference:
    // exact float output, and the output after rounding the weights the way save() does.
    float* Q = allocF(nParams);
    for (size_t i = 0; i < nParams; i++) {
        float scale = i < oW2 ? QA : i < oB2 ? QB : QA * QB;
        Q[i] = std::lround(P[i] * scale) / scale;
    }
    for (size_t i = 0; i < std::min<size_t>(ds.total, 8); i++) {
        process(ds.at(i), P, nullptr, scratch[0]);
        float a = lastOut;
        process(ds.at(i), Q, nullptr, scratch[0]);
        printf("check %zu: float %.1f cp, quantised %.1f cp\n", i, SCALE * a, SCALE * lastOut);
    }
    return 0;
}
