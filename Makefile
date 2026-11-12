CXX      ?= clang++
EXE      ?= halcyon
HIDDEN   ?= 512
LAYOUT   ?= 1
# Path to a network file to embed in the binary. Empty: hand-crafted evaluation.
EVALFILE ?= nets/halcyon-512m-50M.bin

ARCH := $(shell uname -m)
ifeq ($(ARCH),arm64)
  NATIVE := -mcpu=native
else
  NATIVE := -march=native
endif

CXXFLAGS := -std=c++20 -O3 $(NATIVE) -flto -DNDEBUG -Wall -Wextra -pthread \
            -DNNUE_HIDDEN=$(HIDDEN) -DNNUE_LAYOUT=$(LAYOUT)
ifneq ($(EVALFILE),)
  CXXFLAGS += -DEVALFILE=\"$(EVALFILE)\"
endif

SRC := src/main.cpp src/uci.cpp src/search.cpp src/eval.cpp src/nnue.cpp \
       src/position.cpp src/bitboard.cpp src/datagen.cpp
CORE := src/position.cpp src/bitboard.cpp
HDR := $(wildcard src/*.h)

all: $(EXE)

$(EXE): $(SRC) $(HDR) $(EVALFILE)
	$(CXX) $(CXXFLAGS) -o $@ $(SRC)

tools: build/match build/train

build/match: tools/match.cpp $(CORE) $(HDR)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -o $@ tools/match.cpp $(CORE)

build/train: tools/train.cpp $(HDR)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -ffast-math -o $@ tools/train.cpp

build/perft_test: tests/perft_test.cpp $(CORE) $(HDR)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -o $@ tests/perft_test.cpp $(CORE)

test: build/perft_test
	./build/perft_test

clean:
	rm -rf build $(EXE)

.PHONY: all tools test clean
