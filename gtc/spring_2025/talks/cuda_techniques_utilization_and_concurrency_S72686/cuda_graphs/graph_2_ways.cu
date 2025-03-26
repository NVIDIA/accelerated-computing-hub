/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


/* This code example is provided for illustration purposes for GTC 2025 talk:
   CUDA Techniques to Maximize Concurrency and System Utilization [S72686].
   APIs/code may change.
*/


/*
  To compile: nvcc -arch=sm_90 graph_2_ways.cu -o graph_2_ways
  Nsight Systems profile can be collected via: CUDA_VISIBLE_DEVICES=0 CUDA_MODULE_LOADING=EAGER nsys profile --cuda-event-trace=false -o  ./graph_2_ways
*/

#include <iostream>
#include <vector>


#define CUDA_CHECK(expr_to_check) do {            \
    cudaError_t result  = expr_to_check;          \
    if(result != cudaSuccess)                     \
    {                                             \
        fprintf(stderr,                           \
                "CUDA Runtime Error: %s:%i:%d = %s\n", \
                __FILE__,                         \
                __LINE__,                         \
                result,\
                cudaGetErrorString(result));      \
	exit(1);                                  \
    }\
} while(0)


// Reference for %globaltimer: https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-globaltimer-globaltimer-lo-globaltimer-hi
__device__ __forceinline__ unsigned long long __globaltimer()
{
    unsigned long long globaltimer;
    // 64-bit global nanosecond timer
    asm volatile("mov.u64 %0, %globaltimer;"
                 : "=l"(globaltimer));
    return globaltimer;
}


__device__ __forceinline__ void delay_in_us(uint32_t delay_us)
{
    // 64-bit global nanosecond timer
    constexpr uint64_t NS_PER_US = 1000UL;

    uint64_t start_time = __globaltimer();
    uint64_t end_time   = start_time + (delay_us * NS_PER_US);

    while(__globaltimer() < end_time)
    {
    };
}


// delay kernels for illustration purposes
__global__ void kernel_A() {
   delay_in_us(10);
}

__global__ void kernel_B() {
   delay_in_us(20);
}

__global__ void kernel_C() {
   delay_in_us(2);
}

__global__ void kernel_D() {
   delay_in_us(30);
}

__global__ void kernel_E() {
   delay_in_us(50);
}

void stream_example() {
    cudaStream_t strm1, strm2;
    CUDA_CHECK(cudaStreamCreateWithFlags(&strm1, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&strm2, cudaStreamNonBlocking));

    cudaEvent_t event_end_A, event_end_C;
    CUDA_CHECK(cudaEventCreateWithFlags(&event_end_A, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&event_end_C, cudaEventDisableTiming));

    kernel_A<<<1, 32, 0, strm1>>>();
    CUDA_CHECK(cudaGetLastError()); // Reminder: always get and check last error after a kernel launch via <<< >>>, to see if the kernel launch encountered any issues
    CUDA_CHECK(cudaEventRecord(event_end_A, strm1));

    kernel_B<<<1, 32, 0, strm1>>>(); // could have been a separate stream but would then need more dependences
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamWaitEvent(strm2, event_end_A)); // need to make sure event_end_A has already been recorded
    kernel_C<<<1, 32, 0, strm2>>>();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(event_end_C, strm2));

    CUDA_CHECK(cudaStreamWaitEvent(strm1, event_end_C)); // waiting for end of C; waiting for end of B is implicit as B and D are launched in the same stream (strm1).
    kernel_D<<<1, 32, 0, strm1>>>();
    CUDA_CHECK(cudaGetLastError());

    kernel_E<<<1, 32, 0, strm1>>>();
    CUDA_CHECK(cudaGetLastError());

    // Launch again
    kernel_A<<<1, 32, 0, strm1>>>();
    CUDA_CHECK(cudaGetLastError()); // Reminder: always get and check last error after a kernel launch via <<< >>>, to see if the kernel launch encountered any issues
    CUDA_CHECK(cudaEventRecord(event_end_A, strm1));

    kernel_B<<<1, 32, 0, strm1>>>(); // could have been a separate stream but would then need more dependences
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamWaitEvent(strm2, event_end_A)); // need to make sure event_end_A has already been recorded
    kernel_C<<<1, 32, 0, strm2>>>();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(event_end_C, strm2));

    CUDA_CHECK(cudaStreamWaitEvent(strm1, event_end_C)); // waiting for end of C; waiting for end of B is implicit as B and D are launched in the same stream (strm1).
    kernel_D<<<1, 32, 0, strm1>>>();
    CUDA_CHECK(cudaGetLastError());

    kernel_E<<<1, 32, 0, strm1>>>();
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamSynchronize(strm1)); // needed

    // cleanup resources
    CUDA_CHECK(cudaEventDestroy(event_end_A));
    CUDA_CHECK(cudaEventDestroy(event_end_C));
    CUDA_CHECK(cudaStreamDestroy(strm1));
    CUDA_CHECK(cudaStreamDestroy(strm2));
}


void capture_example() {

    cudaGraph_t graph;
    cudaStream_t strm1, strm2;
    CUDA_CHECK(cudaStreamCreateWithFlags(&strm1, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&strm2, cudaStreamNonBlocking));

    cudaEvent_t event_end_A, event_end_C;
    CUDA_CHECK(cudaEventCreateWithFlags(&event_end_A, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&event_end_C, cudaEventDisableTiming));

    CUDA_CHECK(cudaStreamBeginCapture(strm1, cudaStreamCaptureModeGlobal)); // begin stream capture

    kernel_A<<<1, 32, 0, strm1>>>();
    CUDA_CHECK(cudaGetLastError()); //needed because without it if I had 4096 threads in block I wouldn't get kernel1 captured but wouldn't get an error either
    CUDA_CHECK(cudaEventRecord(event_end_A, strm1)); // Side note: could also do cudaEventRecordWithFlags and have cudaEventRecordExternal -> separate event node

    kernel_B<<<1, 32, 0, strm1>>>(); // could have been a separate stream but would then need more dependences
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamWaitEvent(strm2, event_end_A)); // need to make sure event_end_A has already been recorded; could also do option extra 3rd arg cudaEventWaitExternal
    kernel_C<<<1, 32, 0, strm2>>>();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(event_end_C, strm2)); // Side note: could also do cudaEventRecordWithFlags and have cudaEventRecordExternal -> separate event node

    CUDA_CHECK(cudaStreamWaitEvent(strm1, event_end_C)); // waiting for end of C; wait for end of B is implicit given same stream
    kernel_D<<<1, 32, 0, strm1>>>();
    CUDA_CHECK(cudaGetLastError());

    kernel_E<<<1, 32, 0, strm1>>>();
    CUDA_CHECK(cudaGetLastError());
    //CUDA_CHECK(cudaStreamSynchronize(strm1)); // this will give an "operation not permitted when stream is capturing" error
    //CUDA_CHECK(cudaDeviceSynchronize()); // also operation not permitted during stream capture

    CUDA_CHECK(cudaStreamEndCapture(strm1, &graph)); // end stream capture

    // Good practice to check your graph after stream capture
    size_t num_nodes = 0;
    CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &num_nodes));
    printf("nodes in graph %d (stream capture)\n", (int)num_nodes);

    // Uncomment to generate a DOT file, to inspect your graph. If preferable, can convert to pdf as follows:
    // $ dot -Tpdf graph_from_stream_capture -o graph_from_stream_capture.pdf
    // Change flags as needed.
    // CUDA_CHECK(cudaGraphDebugDotPrint(graph, "graph_from_stream_capture", 0 /*cudadaGraphDebugDotFlagsVerbose*/);

    // Instantiate graph
    cudaGraphExec_t graph_exec;
    CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, 0));

    // Launch executable graph twice back to back
    CUDA_CHECK(cudaGraphLaunch(graph_exec, strm1));
    CUDA_CHECK(cudaGraphLaunch(graph_exec, strm1));

    CUDA_CHECK(cudaStreamSynchronize(strm1));

    // cleanup resources
    CUDA_CHECK(cudaEventDestroy(event_end_A));
    CUDA_CHECK(cudaEventDestroy(event_end_C));
    CUDA_CHECK(cudaStreamDestroy(strm1));
    CUDA_CHECK(cudaStreamDestroy(strm2));

    CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
    CUDA_CHECK(cudaGraphDestroy(graph));
}

void set_kernel_params(cudaKernelNodeParams& node_params, void* kernel_function, void** pKernelParams) {
    node_params.blockDim = dim3(32);
    node_params.gridDim  = dim3(1);
    node_params.func     = kernel_function;
    node_params.extra    = nullptr;
    node_params.sharedMemBytes = 0;
    node_params.kernelParams = pKernelParams;
}


void graph_creation_example() {

    cudaGraph_t graph;
    cudaStream_t strm1;
    CUDA_CHECK(cudaStreamCreateWithFlags(&strm1, cudaStreamNonBlocking)); // the stream we will launch the graph on

    cudaGraphNode_t node_A, node_B, node_C, node_D, node_E;
    std::vector<cudaGraphNode_t> node_dependencies;
    cudaKernelNodeParams kernel_node_params[5] = {};
    void* pKArgs[5] = {nullptr};

    void* kernel_functions[5] = {reinterpret_cast<void*>(kernel_A), reinterpret_cast<void*>(kernel_B), reinterpret_cast<void*>(kernel_C), reinterpret_cast<void*>(kernel_D), reinterpret_cast<void*>(kernel_E)};

    for (int i = 0; i < 5; i++) {
        set_kernel_params(kernel_node_params[i], kernel_functions[i], &pKArgs[i]);
    }

    CUDA_CHECK(cudaGraphCreate(&graph, 0)); // Create graph
    CUDA_CHECK(cudaGraphAddKernelNode(&node_A, graph, nullptr /* root node*/, 0 /* no dependencies*/, &kernel_node_params[0]));
    CUDA_CHECK(cudaGraphAddKernelNode(&node_B, graph, &node_A, 1, &kernel_node_params[1]));
    CUDA_CHECK(cudaGraphAddKernelNode(&node_C, graph, &node_A, 1, &kernel_node_params[2]));

    // Node D depends on both node B and node C
    node_dependencies.push_back(node_B);
    node_dependencies.push_back(node_C);
    CUDA_CHECK(cudaGraphAddKernelNode(&node_D, graph, node_dependencies.data(), node_dependencies.size(), &kernel_node_params[3]));

    node_dependencies.clear();
    node_dependencies.push_back(node_D);
    CUDA_CHECK(cudaGraphAddKernelNode(&node_E, graph, node_dependencies.data(), node_dependencies.size(), &kernel_node_params[4]));

    // Check to ensure the graph has 5 nodes
    size_t num_nodes = 0;
    CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &num_nodes));
    printf("nodes in graph %d (graph created using graph APIs)\n", (int)num_nodes);

    // Uncomment to generate a DOT file, to inspect graph. If preferable, can convert to pdf as follows:
    // $ dot -Tpdf graph_with_graph_APIs -o graph_with_graph_APIs.pdf
    // Change flags as needed.
    // CUDA_CHECK(cudaGraphDebugDotPrint(graph, "graph_with_graph_APIs", 0 /*cudadaGraphDebugDotFlagsVerbose*/);

    // Instantiate graph
    cudaGraphExec_t graph_exec;
    CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, 0));

    // Launch the graph twice back to back
    CUDA_CHECK(cudaGraphLaunch(graph_exec, strm1));
    CUDA_CHECK(cudaGraphLaunch(graph_exec, strm1));

    CUDA_CHECK(cudaStreamSynchronize(strm1));

    // cleanup resources
    CUDA_CHECK(cudaStreamDestroy(strm1));
    CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
    CUDA_CHECK(cudaGraphDestroy(graph));
}


int main() {

    /* Examples that express the following:

       kernel_A --------> kernel_C --------|
          |                                |
          |-------------> kernel_B---------v-------> kernel_D --------> kernel_E

       The behavior is demonstrated via:
       - CUDA streams
       - CUDA graph created with stream capture
       - CUDA graph created with graph APIs

       In all cases, the graph/work is executed twice back to back.
    */

    // Launch code using streams and repeat once
    stream_example();

    // Launch code using a CUDA graph created via stream capturing  and repeat once
    capture_example();

    // Launch code using a CUDA graph created via graph APIs and repeat once
    graph_creation_example();

    return 0;
}
