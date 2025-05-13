constexpr int k_block_size = 64; // keep sync overhead low.

constexpr int k_capacity_factor = 3; // somewhat arbitrary
constexpr int k_queue_capacity = k_block_size * k_capacity_factor;

// Process the queue whenever it breaches this size.
// Hence always leave enough room for every thread to possibly insert.
constexpr int k_queue_process_size = k_block_size * (k_capacity_factor - 1);

using Work_item_t = int;

// REQUIRES: 1D blocks. All threads to call, as it calls syncthreads.
// EFFECTS: threads will be synchronized at exit.
// all threads in block work together to clear the queue.
__device__ 
void process_queue(Work_item_t (&block_queue) [k_queue_capacity], int &block_queue_size){
    // assert: block_queue_size <= k_queue_capacity

    for(int queue_idx = threadIdx.x; queue_idx < block_queue_size; queue_idx += blockDim.x){
        const Work_item_t work_item = block_queue[queue_idx];
        // perform deep dive on work item...
    }
    __syncthreads();

    if(threadIdx.x == 0){
        block_queue_size = 0;
    }
    __syncthreads();
}

// REQUIRES: 1D blocks.
__global__ void scout_and_dive_kernel(){

    /*
    Thread does 2 types of work:
        scouting
        deep_dive
    Assume deep dive is expensive enough to be our bottleneck, and the code is highly diverged because not all threads dive at the same time.

    Solution: The deep dive work gets queued to be performed later.
    */

    __shared__ int block_num_threads_finished_scouting;

    // queue of deep dive work.
    __shared__ Work_item_t block_queue[k_queue_capacity];
    __shared__ int block_queue_size;


    // init shared
    if(threadIdx.x == 0){
        block_num_threads_finished_scouting = 0;
        block_queue_size = 0;
    }
    __syncthreads(); // Don't let anyone perform below incr yet.

    const auto & is_thread_finished_scouting = []() { return false; /*your condition here*/};

    if(is_thread_finished_scouting()){
        // deal with any excess of threads launched.
        atomicAdd(&block_num_threads_finished_scouting, 1);
    }
    else {
        // prepare initial scouting work for thread.
    }
    __syncthreads(); // see block_num_threads_finished_executing.


    while(block_num_threads_finished_scouting < k_block_size){

		// invariant: all threads are here. Block is sync'd
        // invariant: some number of threads need to scout
        // invariant: each thread inserts at most one item per iteration.

		// make sure there is enough room in the queue for everyone to insert.
        if(block_queue_size >= k_queue_process_size){
            process_queue(block_queue, block_queue_size);
        }
        // assert: queue size is 0 and block is sync'd

        if(!is_thread_finished_scouting()){
            // perform scouting work...

            const bool found_dive = false; // your condition.
            if(found_dive){
                const Work_item_t work_idx = 0; // your work id mechanism.
                const auto queue_write_dst = atomicAdd(&block_queue_size, 1);
                // assert: queue_write_dst < k_queue_capacity
                block_queue[queue_write_dst] = work_idx;
            }

            // advance to next piece of scouting work
            if(!is_thread_finished_scouting()){
                // advance to next scouting location
            }
            else {
                atomicAdd(&block_num_threads_finished_scouting, 1);
            }

        }

        __syncthreads(); // see block_num_threads_finished_scouting.

    } // while any thread is scouting.

    // all threads done scouting, flush the queue.
    if(block_queue_size > 0){
        process_queue(block_queue, block_queue_size);
    }

}
