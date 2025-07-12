// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime_api.h>
#include <cuda.h>

__global__ void kernel(int *data1, int *data2)
{
  data1[threadIdx.x] = threadIdx.x;
  data2[threadIdx.x] = threadIdx.x;
}

int main()
{
  int *data = (int *)malloc(128 * sizeof(int));

  int data2[128];
  kernel<<<1, 128>>>(data, data2);
  cudaDeviceSynchronize();

  for (int i = 0; i < 128; i++)
  {
    printf("(%d, %d)  ", data[i], data2[i]);
  }
  printf("\n");
  
  free(data);
  return 0;
}

