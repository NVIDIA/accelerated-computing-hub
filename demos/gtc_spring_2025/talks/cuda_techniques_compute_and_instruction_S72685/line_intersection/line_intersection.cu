#include <iostream>
#include <random>
#include <functional>
#include <algorithm>
#include <vector>
#include <chrono>
#include <type_traits>
#include <cstdint>
#include <fstream>
#include <string>


// #define PERFORM_BB_PRECHECK

// #define DIV
// #define DIV_FREE_ND
// #define DIV_FREE_CLRS
static_assert([]() {
    int define_count = 0;
    #ifdef DIV
        ++define_count;
    #endif
    #ifdef DIV_FREE_ND
        ++define_count;
    #endif
    #ifdef DIV_FREE_CLRS
        ++define_count;
    #endif
    return define_count == 1;
}(), "Exactly one of DIV, DIV_FREE_ND, or DIV_FREE_CLRS must be defined!");

constexpr int k_tile_width = 128;
constexpr int k_blockDim_x = 32;
constexpr int k_blockDim_y = 16;

constexpr bool k_print_segs_to_file = false;

using namespace std;

using Scalar_t = float;
using Idx_t = int;
using Intersection_count_t = unsigned int;

struct Vec {
    Scalar_t x;
    Scalar_t y;    

    __host__ __device__
    Vec() { }

    __host__ __device__
    Vec(Scalar_t x_in, Scalar_t y_in)
        : x(x_in), y(y_in){
    }

    __host__ __device__
    Vec operator+(const Vec& rhs) const {
        return Vec(x + rhs.x, y + rhs.y);
    }

    __host__ __device__
    Vec operator-(const Vec& rhs) const {
        return Vec(x - rhs.x, y - rhs.y);
    }

    __host__ __device__
    Scalar_t cross(const Vec &rhs) const {
        return x * rhs.y - y * rhs.x;
    }
};

ostream& operator<<(ostream & os, const Vec &vec){
    os << "(" << vec.x << ", " << vec.y << ")";
    return os;
}

struct Box {
    Vec lower_left;
    Vec upper_right;

    __host__ __device__
    Box(Vec ll_in, Vec ur_in)
        :lower_left(ll_in), upper_right(ur_in) {
    }

    __host__ __device__
    bool contains(const Vec &point) const {
        return lower_left.x <= point.x && point.x <= upper_right.x
            && lower_left.y <= point.y && point.y <= upper_right.y;
    }
    __host__ __device__
    bool overlaps(const Box &other) const {
        return !(
            this->upper_right.x < other.lower_left.x || // this  is left of other.
            other.upper_right.x < this->lower_left.x || // other is left of this.
            this->upper_right.y < other.lower_left.y || // this  is below other
            other.upper_right.y < this->lower_left.y    // other is below this.
        );
    }
};


struct Seg {
    Vec s;
    Vec t;

    __host__ __device__
    Seg() { }

    __host__ __device__
    Seg(Vec s_in, Vec t_in)
        :s(s_in), t(t_in) {
    }

    __device__
    Vec as_vec() const {
        return Vec(t.x - s.x, t.y - s.y);
    }

    __device__
    Box as_box() const {
        return Box( Vec(min(s.x, t.x), min(s.y, t.y)),
                    Vec(max(s.x, t.x), max(s.y, t.y)));
    }

    // The line defined by this segment splits the plane into 2 regions.
    // Returns 0 if point is on the line formed by this segment.
    // Returns positive or negative number depending on which region it lies in.
    __device__
    Scalar_t get_region_of_point(Vec point){
        const Vec s_to_point = point - s;
        return s_to_point.cross(as_vec());
    }
};


ostream& operator<<(ostream & os, const Seg& seg){
    os << seg.s << ", " << seg.t;
    return os;
}

struct CLI_args {
    unsigned int seed;
    Idx_t num_segs;
    float origin_offset_range;
    float length_range;

    CLI_args(int argc, char* argv[]) {
        if (argc != 5) {
            throw std::invalid_argument(
                "Usage: <program_name> <seed> <num_segs> <origin_offset_range> <length_range>");
        }

        seed = std::stoul(argv[1]);
        num_segs = std::stoi(argv[2]);

        static_assert(std::is_same<float, Scalar_t>::value);
        origin_offset_range = std::stof(argv[3]);
        length_range = std::stof(argv[4]);

        cout << "Parsed arguments:\n";
        cout << "\t Seed: " << seed << "\n";
        cout << "\t Num Segs: " << num_segs << "\n";
        cout << "\t Origin Offset Range: " << origin_offset_range << "\n";
        cout << "\t Length Range: " << length_range << "\n";
    }
};

vector<Seg> generate_random_segs(CLI_args args){

    static_assert(std::is_same<float, Scalar_t>::value);
    std::default_random_engine origin_engine(args.seed);
    std::default_random_engine length_engine(args.seed);

    std::uniform_real_distribution<Scalar_t> origin_dist(-args.origin_offset_range, args.origin_offset_range);
    std::uniform_real_distribution<Scalar_t> length_dist(-args.length_range, args.length_range);

    auto origin_dice = std::bind(origin_dist, origin_engine);
    auto length_dice = std::bind(length_dist, length_engine);

    vector<Seg> segs(args.num_segs);

    for(auto& seg : segs){
        Vec origin = Vec(origin_dice(), origin_dice());
        Vec s_diff = Vec(length_dice(), length_dice());
        Vec t_diff = Vec(length_dice(), length_dice());
        seg = Seg(origin + s_diff, origin + t_diff);
    }

    return segs;

}


__device__ 
bool do_segs_intersect_divFreeCLRS(Seg seg1, Seg seg2){

    // if s1_reg and t1_reg are both non-zero and differ in sign,
        // then seg 1 straddles the LINE formed by seg 2
    const Scalar_t s1_reg = seg2.get_region_of_point(seg1.s);
    const Scalar_t t1_reg = seg2.get_region_of_point(seg1.t);
    // if s2_reg and t2_reg are both non-zero and differ in sign,
        // then seg 2 straddles the LINE formed by seg 1.
    const Scalar_t s2_reg = seg1.get_region_of_point(seg2.s);
    const Scalar_t t2_reg = seg1.get_region_of_point(seg2.t);

    const bool any_zero_region =
        (s1_reg == 0 || t1_reg == 0 || s2_reg == 0 || t2_reg == 0);

    // Check for intersections s.t. the intersection point IS NOT ANY END POINT.
    if(!any_zero_region
            && s1_reg < 0 != t1_reg < 0 // regions must differ
            && s2_reg < 0 != t2_reg < 0){
        return true; 
    }

    // Check for intersections s.t. the intersection point IS ANY ENDPOINT.
    if(s1_reg == 0 && seg2.as_box().contains(seg1.s)){
        return true;  // seg1_s on seg2
    }
    if(t1_reg == 0 && seg2.as_box().contains(seg1.t)){
        return true;  // seg1_t on seg 2
    }
    if(s2_reg == 0 && seg1.as_box().contains(seg2.s)){
        return true;  // seg2_s on seg 1
    }
    if(t2_reg == 0 && seg1.as_box().contains(seg2.t)){
        return true;  // seg2_t on seg 1
    }

    // no intersection.
    return false;
}

__device__ __forceinline__
bool incl01(Scalar_t num, Scalar_t den){

    if(abs(num) < 1.0e-7f){
        return true;
    }

    const bool num_positive = num >= 0;
    const bool den_positive = den >= 0;

    const bool gt = num > den;
    const bool lt = num < den;
    return !(
        (num_positive != den_positive) |
        (num_positive & gt) | // both have positive sign, numerator larger in magnitude
        ((!num_positive) & lt) // both have negative sign, numerator larger in magnitude.
    );
}

template<typename T>
__device__
bool incl01(T v) {
    return T(0) <= v && v <= T(1);
}

__device__
bool do_segs_intersect_INCL(Seg seg1, Seg seg2){

    const Vec vec1 = seg1.as_vec();
    const Vec vec2 = seg2.as_vec();

    const Scalar_t vec_cross = vec1.cross(vec2);

    if(vec_cross != 0){
        const Vec s1_to_s2 = seg2.s - seg1.s;
        const Scalar_t l_num = s1_to_s2.cross(vec2);
        const Scalar_t m_num = s1_to_s2.cross(vec1);

        #ifdef DIV
            const Scalar_t l = l_num / vec_cross;
            const Scalar_t m = m_num / vec_cross;
            return incl01(l) && incl01(m);
        #endif

        #ifdef DIV_FREE_ND
            return incl01(l_num, vec_cross) && incl01(m_num, vec_cross);
        #endif
    }
    return false;
}


__global__
void count_intersections_kernel(Seg *segs, Idx_t num_segs, Intersection_count_t *num_intersections){

    // Visualization of possible intersections.
    // Each X is a seg-seg pair to be tested.
    //   0 1 2 3 4 5
    // 0 . X X X X X
    // 1 . . X X X X
    // 2 . . . X X X
    // 3 . . . . X X
    // 4 . . . . . X
    // 5 . . . . . .

    const Idx_t block_first_col = blockIdx.x * k_tile_width;
    const Idx_t block_first_row = blockIdx.y * k_tile_width;

    {
        const int block_closest_to_diag_col = block_first_col + (k_tile_width - 1);
        const int block_closest_to_diag_row = block_first_row;
        // We need col > row to do work.
        const bool are_all_elems_below_diag = block_closest_to_diag_col <= block_closest_to_diag_row;
        if(are_all_elems_below_diag){
            return;
        }
    }

    __shared__ Seg tile_col_segs[k_tile_width];
    __shared__ Seg tile_row_segs[k_tile_width];
    __shared__ Intersection_count_t block_num_intersections;

    if(threadIdx.x == 0 && threadIdx.y == 0){
        block_num_intersections = 0;
    }

    const auto intra_halfBlock_tid = (threadIdx.y / 2) * blockDim.x + threadIdx.x;
    const auto threads_per_half_block = blockDim.x * (blockDim.y / 2);
    if(threadIdx.y % 2 == 0){
        for(Idx_t tile_col = intra_halfBlock_tid; tile_col < k_tile_width; tile_col += threads_per_half_block){
            const Idx_t col = block_first_col + tile_col;
            if(col >= num_segs){
                break;
            }
            tile_col_segs[tile_col] = segs[col];
        }
    }
    else {
        for(Idx_t tile_row = intra_halfBlock_tid; tile_row < k_tile_width; tile_row += threads_per_half_block){
            const Idx_t row = block_first_row + tile_row;
            if(row >= num_segs){
                break;
            }
            tile_row_segs[tile_row] = segs[row];
        }
    }

    __syncthreads();

    for(Idx_t tile_col = threadIdx.x; tile_col < k_tile_width; tile_col += blockDim.x){
        const Idx_t col = block_first_col + tile_col;
        if(col >= num_segs){
            break;
        }
        const Seg seg1 = tile_col_segs[tile_col];

        for(Idx_t tile_row = threadIdx.y; tile_row < k_tile_width; tile_row += blockDim.y){
            const Idx_t row = block_first_row + tile_row;
            if(row >= num_segs){
                break;
            }
            if(row >= col){
                continue; // we are on or below the diagonal. Continue.
            }

            const Seg seg2 = tile_row_segs[tile_row];

            #ifdef PERFORM_BB_PRECHECK
                const bool promising = seg1.as_box().overlaps(seg2.as_box());
            #else
                const bool promising = true;
            #endif

            if(promising){

                #ifdef DIV_FREE_CLRS
                    const bool intersects = do_segs_intersect_divFreeCLRS(seg1, seg2);
                #else
                    const bool intersects = do_segs_intersect_INCL(seg1, seg2);
                #endif

                if(intersects){
                    atomicAdd(&block_num_intersections, static_cast<Intersection_count_t>(1));
                }
            }

        }
    }


    __syncthreads();
    if(threadIdx.x == 0 && threadIdx.y == 0){
        atomicAdd(num_intersections, block_num_intersections);
    }
}

size_t div_round_up(size_t num, size_t den){
    return (num + (den - 1)) / (den);
}

int count_intersections(const vector<Seg> &segs){

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    Seg *d_segs;
    const auto segs_bytes = sizeof(Seg) * segs.size();
    cudaMallocAsync(&d_segs, segs_bytes, stream);
    cudaMemcpyAsync(d_segs, segs.data(), segs_bytes, cudaMemcpyHostToDevice, stream);

    Intersection_count_t *d_num_intersections;
    cudaMallocAsync(&d_num_intersections, sizeof(Intersection_count_t), stream);
    cudaMemsetAsync(d_num_intersections, 0, sizeof(Intersection_count_t), stream);

    dim3 block_size(k_blockDim_x, k_blockDim_y);
    dim3 grid_size;
    grid_size.x = div_round_up(segs.size(), k_tile_width);
    grid_size.y = div_round_up(segs.size(), k_tile_width);
    // Not concerning ourselves with the waste produced by launching blocks that get mapped to the lower left of that seg-seg matrix.

    count_intersections_kernel<<<grid_size, block_size, 0, stream>>>(d_segs, segs.size(), d_num_intersections);

    Intersection_count_t num_intersections;
    cudaMemcpyAsync(&num_intersections, d_num_intersections, sizeof(Intersection_count_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFreeAsync(d_num_intersections, stream);
    cudaFreeAsync(d_segs, stream);
    cudaStreamDestroy(stream);
    return num_intersections;
}

void print(std::ostream& os, const vector<Seg> & segs, Intersection_count_t num_intersections){
    os << "segs: " << endl;
    for(const auto& seg : segs){
        os << seg << endl;
    }
    os << "Number of intersection: " << num_intersections << endl;

}

// returns computed number of intersections.
Intersection_count_t test_instance(string test_id_str, const vector<Seg> & segs, Intersection_count_t expected_num_intersections){

    const Intersection_count_t num_intersections = count_intersections(segs);

    if(num_intersections == expected_num_intersections || test_id_str == "perf"){
        cout << "Passed test " << test_id_str << endl;
    }
    else {
        throw std::runtime_error("Failed test " + test_id_str);
    }

    if(k_print_segs_to_file){
        string filename = "test_" + test_id_str + ".txt";
        ofstream fout(filename);
        if(!fout.is_open()){
            throw std::runtime_error("Could not open " + filename + " for writing.");
        }
        print(fout, segs, num_intersections);

        fout.close();
    }

    return num_intersections;
}

double nChoose2(double n) {
    if (n < 2) {
        return 0; // If n < 2, no valid combinations exist
    }
    return (n * (n - 1)) / 2; // Formula for C(n, 2)
}

void test_correctness(){
    vector<Seg> segs;

    // test 1
    segs.clear();
    segs.push_back(Seg(Vec(-3, -3), Vec(3, 3)));
    segs.push_back(Seg(Vec(-4.5, -3.4), Vec(-2.3, -5))); // does not intersect first
    segs.push_back(Seg(Vec(-6.1, -2.4), Vec(5.7, -6))); // barely does not intersect.
    segs.push_back(Seg(Vec(-3.3, -0.2), Vec(0, -1.7))); // intersects
    segs.push_back(Seg(Vec(0.4, -0.6), Vec(1.2, -1.2))); // does not intersect
    segs.push_back(Seg(Vec(0.1, 1.6), Vec(0.9, 1.9))); // does not intersect
    test_instance("1", segs, 1);

    // test 2
    segs.push_back(Seg(Vec(1.15, -0.778), Vec(-5.95, -2.9)));
    test_instance("2", segs, 1 + 4);
}

void test_perf(CLI_args args){

    // seed = std::chrono::system_clock::now().time_since_epoch().count();
    vector<Seg> segs = generate_random_segs(args);
    Intersection_count_t num_intersections = test_instance("perf", segs, 0);

    double num_possible_intersections = nChoose2(args.num_segs);
    cout << "Perf: Number of possible intersections: " << num_possible_intersections << endl;
    cout << "Perf: Number of intersection: " << static_cast<double>(num_intersections) << endl;
    cout << "Perf: Percentage intersections: " << num_intersections / num_possible_intersections * 100 << " % " << endl;
}

int main(int argc, char* argv[]){

    CLI_args args(argc, argv);

    // test_correctness();
    test_perf(args);

    return 0;
}
