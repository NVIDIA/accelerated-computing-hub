#pragma once

namespace tutorial {

    typedef enum {
        EMU_MATMUL_TILE_128x128x128 = 0,
        EMU_MATMUL_TILE_256x128x128 = 1,
        EMU_MATMUL_TILE_128x256x128 = 2,
        EMU_MATMUL_TILE_128x128x256 = 3,
    } emuMatmulTile_t;

    typedef enum {
        EMU_MATMUL_STAGES_1 = 0,
        EMU_MATMUL_STAGES_2 = 1,
        EMU_MATMUL_STAGES_4 = 2,
        EMU_MATMUL_STAGES_8 = 3,
    } emuMatmulStages_t;

    typedef enum {
        EMU_BLOCK_DIM_64x1x1 = 0,
        EMU_BLOCK_DIM_128x1x1 = 1,
        EMU_BLOCK_DIM_256x1x1 = 2
    } emuMatmulBlockDim_t;


template <emuMatmulTile_t T>
struct EmuTileInfo;

#define MAKE_EMU_TILE_INFO(TILE_M_, TILE_N_, TILE_K_)                   \
    template <>                                                         \
    struct EmuTileInfo<EMU_MATMUL_TILE_##TILE_M_##x##TILE_N_##x##TILE_K_> {  \
        static constexpr int TILE_M = TILE_M_;                          \
        static constexpr int TILE_N = TILE_N_;                          \
        static constexpr int TILE_K = TILE_K_;                          \
    }

MAKE_EMU_TILE_INFO(128, 128, 128);
MAKE_EMU_TILE_INFO(256, 128, 128);
MAKE_EMU_TILE_INFO(128, 256, 128);
MAKE_EMU_TILE_INFO(128, 128, 256);
    
template <emuMatmulStages_t T>
struct EmuStageInfo;

#define MAKE_EMU_STAGE_INFO(STAGE_COUNT_)                  \
    template <>                                            \
    struct EmuStageInfo<EMU_MATMUL_STAGES_##STAGE_COUNT_> { \
        static constexpr int STAGE_COUNT = STAGE_COUNT_;   \
    }

MAKE_EMU_STAGE_INFO(1);
MAKE_EMU_STAGE_INFO(2);
MAKE_EMU_STAGE_INFO(4);
MAKE_EMU_STAGE_INFO(8);

template <emuMatmulBlockDim_t T>
struct EmuBlockDimInfo;

#define MAKE_EMU_BLOCK_DIM_INFO(BLOCK_X_, BLOCK_Y_, BLOCK_Z_)             \
    template <>                                                           \
    struct EmuBlockDimInfo<EMU_BLOCK_DIM_##BLOCK_X_##x##BLOCK_Y_##x##BLOCK_Z_> { \
        static constexpr int BLOCK_X = BLOCK_X_;                           \
        static constexpr int BLOCK_Y = BLOCK_Y_;                           \
        static constexpr int BLOCK_Z = BLOCK_Z_;                           \
    }

MAKE_EMU_BLOCK_DIM_INFO(64, 1, 1);
MAKE_EMU_BLOCK_DIM_INFO(128, 1, 1);
MAKE_EMU_BLOCK_DIM_INFO(256, 1, 1);

}
