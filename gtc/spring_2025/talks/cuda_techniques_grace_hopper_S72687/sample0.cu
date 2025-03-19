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

#include <cstdlib>

#include <cuda_runtime_api.h>
#include <cuda.h>

__global__ void kernel(int *data)
{
    data[threadIdx.x] += 2;
}
int main()
{
    int N = 128;
    int *data = (int *)malloc(N * sizeof(int));
    int *d_data;

    cudaMalloc((void **)&d_data, N * sizeof(int));

    cudaMemcpy(d_data, data, N * sizeof(int), cudaMemcpyHostToDevice);

    kernel<<<1, N>>>(d_data);

    cudaMemcpy(d_data, data, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(d_data);
    free(data);
}

