
#include <cuda/std/mdspan>
#include <cuda/std/array>
#include <cstdio>

int main() {
  cuda::std::array<int, 6> sd {0, 1, 2, 3, 4, 5};
  cuda::std::mdspan md(sd.data(), 2, 3);

  std::printf("md(0, 0) = %d\n", md(0, 0)); // 0
  std::printf("md(1, 2) = %d\n", md(1, 2)); // 5

  std::printf("size   = %zu\n", md.size());    // 6
  std::printf("height = %zu\n", md.extent(0)); // 2
  std::printf("width  = %zu\n", md.extent(1)); // 3
}
