#pragma once

#include "numerical.hpp"

namespace tutorial {

    enum class random_distribution
    {
        any,
        uniform,
        normal
    };

    template<typename T, class Processor, class Dist>
    std::vector<T> get_random_vector(Processor const& proc, Dist& dist, const size_t size, int seed = -1) {

        std::vector<T> ret(size);

        std::generate(ret.begin(), ret.end(), [&]() {
            static thread_local std::random_device rd;
            static thread_local std::ranlux24_base gen((seed != -1) ? seed : rd());

            return convert<T>(proc(dist(gen)));
        });

        return ret;
    }

    template<random_distribution Distribution, typename ValueType>
    struct random_generator {
        template<class... OptionalDistArgs>
        static std::vector<ValueType> generate(size_t size, OptionalDistArgs... dist_args);
    };

    template<typename ValueType>
    struct random_generator<random_distribution::normal, ValueType> {
        static std::vector<ValueType> generate(size_t size, float mean, float sd, const int seed = -1) {
            static_assert(commondx::is_floating_point_v<ValueType>, "Floating point output type required");
            auto dist = std::normal_distribution<float>(mean, sd);
            auto proc = cublasdx::identity {};
            return get_random_vector<ValueType>(proc, dist, size, seed);
        }
    };

    template<typename ValueType>
    struct random_generator<random_distribution::uniform, ValueType> {
        template<typename MinMaxType>
        static std::vector<ValueType> generate(size_t size, MinMaxType min, MinMaxType max, int seed = -1) {
            static_assert(commondx::is_floating_point_v<ValueType> or commondx::is_integral_v<ValueType>,
                          "Datatype must be either recognized floating point or integral");
            auto dist = [&]() {
                if constexpr (commondx::is_floating_point_v<ValueType>) {
                    return std::uniform_real_distribution<double>(min, max);
                } else {
                    return std::uniform_int_distribution<int32_t>(min, max);
                }
                CUTE_GCC_UNREACHABLE;
            }();
            auto proc = cublasdx::identity {};
            return get_random_vector<ValueType>(proc, dist, size, seed);
        }
    };

    template<typename ValueType>
    struct random_generator<random_distribution::any, ValueType> {
        static std::vector<ValueType> generate(size_t size, int seed = -1) {
            if constexpr (commondx::is_floating_point_v<ValueType>) {
                return random_generator<random_distribution::normal, ValueType>::generate(size, 0.0, 1.0, seed);
            } else if constexpr (commondx::is_signed_integral_v<ValueType>) {
                return random_generator<random_distribution::uniform, ValueType>::generate(size, -20, 20, seed);
            } else if constexpr (commondx::is_unsigned_integral_v<ValueType>) {
                return random_generator<random_distribution::uniform, ValueType>::generate(size, 0, 40, seed);
            } else {
                static_assert(commondx::is_floating_point_v<ValueType> or commondx::is_integral_v<ValueType>);
            }
            CUTE_GCC_UNREACHABLE;
        }
    };

    template<class ValueType,
             cublasdx::arrangement Arrangement,
             random_distribution   RandomDistribution = random_distribution::any,
             class... OptionalDistArgs>
    auto get_random_device_tensor(int size_x, int size_y, OptionalDistArgs... optional_dist_args) {
        std::vector random_host_data =
            random_generator<RandomDistribution, ValueType>::generate(size_x * size_y, optional_dist_args...);

        thrust::device_vector<ValueType> device_vector = random_host_data;
        auto                             iter  = cute::make_gmem_ptr(thrust::raw_pointer_cast(device_vector.data()));
        auto                             shape = cute::make_shape(size_x, size_y);
        auto stride_atom = cute::conditional_return<Arrangement == cublasdx::arrangement::col_major>(
            cute::LayoutLeft {}, cute::LayoutRight {});
        auto tensor = cute::make_tensor(iter, shape, stride_atom);

        return cuda::std::make_tuple(std::move(device_vector), tensor);
    }

    template<class Tensor>
    __global__ void make_tensor_symmetrix(Tensor tensor, matrix_half data_half) {
        auto tid_m = threadIdx.x + blockIdx.x * blockDim.x;
        auto tid_n = threadIdx.y + blockIdx.y * blockDim.y;

        if (tid_n > tid_m)
            return;
    }

    template<class ValueType,
             cublasdx::arrangement Arrangement,
             random_distribution   RandomDistribution = random_distribution::any,
             class... OptionalDistArgs>
    auto get_symmetric_random_device_tensor(int         side_length,
                                            matrix_half data_half,
                                            OptionalDistArgs... optional_dist_args) {
        int         num_unique_elems = (side_length * (side_length + 1)) / 2;
        std::vector random_host_data =
            random_generator<RandomDistribution, ValueType>::generate(num_unique_elems, optional_dist_args...);

        std::vector<ValueType> symm_host_data(side_length * side_length);

        auto get_linear_index = [&](int x, int y) {
            bool is_reversed =
                (data_half == matrix_half::lower and y > x) or (data_half == matrix_half::upper and x > y);

            int actual_x = is_reversed ? x : y;
            int actual_y = is_reversed ? y : x;

            auto const k_lower = ((actual_x - 1) * actual_x) / 2 + actual_y;
            auto const k_upper = ((2 * side_length - actual_x) * (actual_x + 1)) / 2 - (side_length - actual_y);

            return (data_half == matrix_half::lower) ? k_lower : k_upper;
        };

        auto shape       = cute::make_shape(side_length, side_length);
        auto stride_atom = cute::conditional_return<Arrangement == cublasdx::arrangement::col_major>(
            cute::LayoutLeft {}, cute::LayoutRight {});
        auto layout = cute::make_layout(shape, stride_atom);

        auto host_tensor = cute::make_tensor(symm_host_data.data(), layout);

        for (int i = 0; i < side_length; ++i) {
            for (int j = 0; j < side_length; ++j) {
                auto data_idx     = get_linear_index(i, j);
                host_tensor(i, j) = random_host_data.at(data_idx);
            }
        }

        thrust::device_vector<ValueType> device_vector = symm_host_data;
        auto                             iter   = cute::make_gmem_ptr(thrust::raw_pointer_cast(device_vector.data()));
        auto                             tensor = cute::make_tensor(iter, layout);

        return cuda::std::make_tuple(std::move(device_vector), tensor);
    }

} // namespace tutorial
