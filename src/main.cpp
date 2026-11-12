#include "search.h"

void uciLoop(int argc, char** argv);

int main(int argc, char** argv) {
    initBitboards();
    initZobrist();
    initHCE();
    initSearch();
    NNUE::loadEmbedded();
    uciLoop(argc, argv);
    return 0;
}
