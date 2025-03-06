// Static block size:

// gridDim is 1D. One block per polygon.
// blockDim is 3D. Sized arbitrarily.
const auto polygon_idx = blockIdx.x;

// fetch characteristics of polygon.
const auto x_size = ...;
const auto y_size = ...;
const auto edges_size = ...;

for (int x_idx = threadIdx.x; x_idx < x_size; x_idx += blockDim.x) {
    for (int y_idx = threadIdx.y; y_idx < y_size; y_idx += blockDim.y) {
        for (int edge_idx = threadIdx.z; edge_idx < edges_size; edge_idx += blockDim.z) {
            // work...
        }
    }
}

// =============================================
// =============================================
// =============================================

// Dynamic block size:
// More efficient use of threads when polygon characteristics vary widely.

// gridDim is 1D. One block per polygon.
// blockDim is 1D. Size is a power of 2.
const auto polygon_idx = blockIdx.x;

// fetch characteristics of polygon.
const auto x_size = ...;
const auto y_size = ...;
const auto edges_size = ...;

// Determine virtual blockDim. Assume concrete block size is 1D and a power of 2.
dim3 vBlockDim(1, 1, 1);
const auto threads_allocated = 1;

// first fill X​
while(vBlockDim.x < x_size && threads_allocated < blockDim.x) {
    vBlockDim.x *= 2;
    threads_allocated *= 2;
}
// then fill Y​
while(vBlockDim.y < y_size && threads_allocated < blockDim.x) {
    vBlockDim.y *= 2;
    threads_allocated *= 2;
}
// then fill E
while(vBlockDim.z < edges_size && threads_allocated < blockDim.x) {
    vBlockDim.z *= 2;
    threads_allocated *= 2;
}

// Determine virtual threadIdx
dim3 vThreadIdx;
vThreadIdx.x = threadIdx.x % vBlockDim.x;
vThreadIdx.y = (threadIdx.x / vBlockDim.x) % vBlockDim.y;
vThreadIdx.z = (threadIdx.x / (vBlockDim.x * vBlockDim.y));

// work
for (int x_idx = vThreadIdx.x; x_idx < x_size; x_idx += vBlockDim.x) {
    for (int y_idx = vThreadIdx.y; y_idx < y_size; y_idx += vBlockDim.y) {
        for (int edge_idx = vThreadIdx.z; edge_idx < edges_size; edge_idx += vBlockDim.z) {
            // work...
        }
    }
}

