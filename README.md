# Halcyon

A UCI chess engine written from scratch in C++20 with no dependencies. The engine, the match
runner used to test it, the training-data generator and the neural-network trainer all live in
this repository; the network is trained only on the engine's own self-play games.

## Build

```
make                       # builds ./halcyon with the network in nets/ embedded
make tools                 # builds build/match (match runner) and build/train (trainer)
make test                  # move-generator validation (perft)
```

`make EVALFILE=` builds without a network (hand-crafted evaluation). `HIDDEN` and `LAYOUT` must
match the network file being embedded.

Two networks trained on 50M self-play positions are included:

| File | Build flags | Notes |
|---|---|---|
| `nets/halcyon-512m-50M.bin` | default (`HIDDEN=512 LAYOUT=1`) | 512-wide, horizontally mirrored |
| `nets/halcyon-256-50M.bin` | `HIDDEN=256 LAYOUT=0 EVALFILE=nets/halcyon-256-50M.bin` | smaller and faster; about equal at very fast time controls |

Use `./halcyon` in any UCI chess GUI (Cute Chess, Banksia, En Croissant, Arena...).
UCI options: `Hash` (MB), `Threads`, `MoveOverhead` (ms), `EvalFile`.

## Strength

Estimated rating: **about 3240** (likely range 3215–3265), on Stockfish's limited-strength scale.

Measured with the default build (512-wide network, 50M positions) against Stockfish 19 with
`UCI_LimitStrength=true` and `UCI_Elo=3190`, on an Apple M5:

| | |
|---|---|
| Time control | 2 minutes + 1 second per move |
| Games | 300, from `books/hce3000.epd`, each opening played twice with colours reversed |
| Result for Halcyon | +103 −60 =137 (score 57.2%) |
| Elo difference | +50 ±25 (95% confidence) |
| Time losses | 2 (the match runner does not record which engine lost them) |

The rating is 3190 plus the measured difference. It is only as accurate as Stockfish's
`UCI_Elo` setting, which is Stockfish's own calibration rather than a rating from an official list.

To reproduce:

```
./build/match -e1 ./halcyon -e2 /opt/homebrew/bin/stockfish \
  -o2 "UCI_LimitStrength=true;UCI_Elo=3190" \
  -book books/hce3000.epd -tc 120+1 -games 300 -c 9 -hash 128
```

## Design

| Part | Implementation |
|---|---|
| Board | Bitboards, magic-bitboard sliders, copy-make, staged pseudo-legal generation with a legality check |
| Search | Iterative deepening PVS, aspiration windows, quiescence search, Lazy SMP |
| Pruning | Null move (with verification), reverse futility, razoring, ProbCut, late-move pruning, futility, SEE and history pruning |
| Extensions / reductions | Singular extensions (double/triple, negative), multi-cut, late-move reductions, internal iterative reductions |
| Move ordering | TT move, capture history + MVV, killer, threat-aware butterfly history, continuation history (1, 2, 4 plies), pawn-structure history |
| Eval correction | Correction history keyed by pawn, non-pawn, minor, major and continuation |
| Evaluation | NNUE `(768 x king buckets -> N) x 2 -> 1 x 8` with SCReLU, incremental accumulators, per-bucket refresh cache, NEON inference |
| Hash table | 3-entry buckets, 10-byte entries, depth/age replacement, shared lock-free between threads |

## Source layout

```
src/bitboard.*   attack tables and magic bitboards
src/position.*   board state, make move, legality, SEE, move generation, perft
src/search.*     search, move ordering, history heuristics, time management
src/nnue.*       network inference; nnue_arch.h is shared with the trainer
src/eval.cpp     hand-crafted evaluation (bootstrap only)
src/datagen.*    self-play data generation and opening generation
src/uci.cpp      UCI protocol and extra commands
tools/match.cpp  engine-vs-engine match runner with SPRT
tools/train.cpp  multi-threaded CPU NNUE trainer
tests/           perft and move-validation tests
```

## Development loop

```
./halcyon genfens 3000 7 books/openings.epd                 # balanced random openings
./build/match -e1 ./halcyon -e2 ./build/base -book books/openings.epd -tc 4+0.04 -sprt 0 5 -c 8
./halcyon datagen 10 50000000 5000 data/gen.bin             # threads, positions, nodes per move
./build/train -data data/gen.bin -out nets/new.bin -hidden 256 -layout 0 -epochs 30
```

Other commands: `bench [depth]`, `perft <depth>`, `eval`, `checknet <data> <n>`.
