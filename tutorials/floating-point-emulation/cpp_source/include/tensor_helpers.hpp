#pragma once

// For CuTe Tensor types
#include <cublasdx.hpp>

// For required cuda::std types
#include <cuda/std/tuple>
#include <cuda/std/utility>
#include <cuda/std/type_traits>

namespace tutorial {

    template<class Tensor>
    struct tensor_value_type;

    template<class Engine, class Layout>
    struct tensor_value_type<cublasdx::tensor<Engine, Layout>> {
        using type = typename Engine::value_type;
    };

    template<class T>
    using tensor_value_type_t = typename tensor_value_type<T>::type;


    namespace detail {

        template<class Element>
        CUBLASDX_HOST_DEVICE auto convert_to_cute_tuple_element(Element const& elem) {
            static_assert(cuda::std::is_integral_v<Element>, "Only flat integral tuples are supported");
            return elem;
        }

        template<class Element, Element Value>
        CUBLASDX_HOST_DEVICE auto convert_to_cute_tuple_element(cuda::std::integral_constant<Element, Value>) {
            static_assert(cuda::std::is_integral_v<Element>, "Only flat integral tuples are supported");
            return cute::Int<Value> {};
        }

        template<auto Value>
        CUBLASDX_HOST_DEVICE auto convert_to_cute_tuple_element(cute::C<Value>) {
            static_assert(cuda::std::is_integral_v<decltype(Value)>, "Only flat integral tuples are supported");
            return cute::Int<Value> {};
        }

        template<class... TupleArgs, int... Indices>
        CUBLASDX_HOST_DEVICE auto convert_to_cute_tuple(cuda::std::tuple<TupleArgs...> const& std_tuple,
                                                        cuda::std::integer_sequence<int, Indices...>) {
            return cute::make_tuple(convert_to_cute_tuple_element(cuda::std::get<Indices>(std_tuple))...);
        }

        template<class... TupleArgs>
        CUBLASDX_HOST_DEVICE auto convert_to_cute_tuple(cuda::std::tuple<TupleArgs...> const& std_tuple) {
            constexpr unsigned num_elems = sizeof...(TupleArgs);
            return convert_to_cute_tuple(std_tuple, cuda::std::make_integer_sequence<int, num_elems>());
        }
    } // namespace detail

    template<class PointerType, class... ShapeArgs, class... StrideArgs>
    CUBLASDX_HOST_DEVICE auto make_gmem_tensor_from_tuples(PointerType*                           pointer_type,
                                                           cuda::std::tuple<ShapeArgs...> const&  shape,
                                                           cuda::std::tuple<StrideArgs...> const& stride) {

        auto cute_shape  = detail::convert_to_cute_tuple(shape);
        auto cute_stride = detail::convert_to_cute_tuple(stride);
        auto cute_layout = cute::make_layout(cute_shape, cute_stride);

        return cute::make_tensor(cute::make_gmem_ptr(pointer_type), cute_layout);
    }

    template<class... ShapeArgs, class... StrideArgs>
    CUBLASDX_HOST_DEVICE auto make_layout_from_tuples(cuda::std::tuple<ShapeArgs...> const&  shape,
                                                      cuda::std::tuple<StrideArgs...> const& stride) {

        auto cute_shape  = detail::convert_to_cute_tuple(shape);
        auto cute_stride = detail::convert_to_cute_tuple(stride);
        return cute::make_layout(cute_shape, cute_stride);
    }

    template<class T>
    struct is_integral: cuda::std::is_integral<T> {};

    template<int N>
    struct is_integral<cuda::std::integral_constant<int, N>>: cuda::std::true_type {};

    template<class T>
    inline constexpr bool is_integral_v = is_integral<T>::value;

    template<cublasdx::arrangement Arr, int... Ints>
    constexpr auto make_order(cuda::std::integer_sequence<int, 0, 1, Ints...>) {
        auto col_major_order = cute::Step<cute::_0, cute::_1, cute::Int<Ints>...> {};
        auto row_major_order = cute::Step<cute::_1, cute::_0, cute::Int<Ints>...> {};

        return cute::conditional_return<Arr == cublasdx::arrangement::col_major>(col_major_order, row_major_order);
    }

    template<class ValueType, cublasdx::arrangement Arrangement, class... Dimensions>
    auto get_empty_device_tensor(Dimensions... dimensions) {

        static_assert((is_integral_v<Dimensions> && ...));
        auto const                       total_size = (dimensions * ...);
        thrust::device_vector<ValueType> device_vector(total_size);

        auto       iter  = cute::make_gmem_ptr(thrust::raw_pointer_cast(device_vector.data()));
        auto const shape = cute::make_shape(detail::convert_to_cute_tuple_element(dimensions)...);
        auto const stride_atom =
            make_order<Arrangement>(cuda::std::make_integer_sequence<int, sizeof...(Dimensions)>());
        auto const layout = cute::make_ordered_layout(shape, stride_atom);
        auto       tensor = cute::make_tensor(iter, layout);

        return cuda::std::make_tuple(std::move(device_vector), tensor);
    }

    template<cublasdx::arrangement Arrangement, class PointerType, class... ShapeArgs>
    CUBLASDX_HOST_DEVICE auto make_gmem_tensor_from_tuples(PointerType*                          pointer_type,
                                                           cuda::std::tuple<ShapeArgs...> const& shape) {

        auto cute_shape = detail::convert_to_cute_tuple(shape);

        auto const stride_atom =
            make_order<Arrangement>(cuda::std::make_integer_sequence<int, cute::rank(decltype(cute_shape) {})>());
        auto const layout = cute::make_ordered_layout(cute_shape, stride_atom);

        return cute::make_tensor(cute::make_gmem_ptr(pointer_type), layout);
    }

    template<cublasdx::arrangement Arrangement, int SizeX, int SizeY, class Iterator>
    __host__ __device__ __forceinline__ auto make_smem_tensor(Iterator* iterator) {
        auto iter        = cute::make_smem_ptr(iterator);
        auto shape       = cute::Shape<cute::Int<SizeX>, cute::Int<SizeY>> {};
        auto stride_atom = cute::conditional_return<Arrangement == cublasdx::arrangement::col_major>(
            cute::LayoutLeft {}, cute::LayoutRight {});

        return cute::make_tensor(iter, cute::make_layout(shape, stride_atom));
    }

    template<class Tensor>
    auto get_copy_tensor(Tensor old_tensor) {
        using tensor_value_type                               = tensor_value_type_t<Tensor>;
        auto                                     tensor_elems = size(old_tensor.layout());
        thrust::device_vector<tensor_value_type> device_vector(tensor_elems);
        CUDA_CHECK_AND_EXIT(cudaMemcpy(thrust::raw_pointer_cast(device_vector.data()),
                                       raw_pointer_cast(old_tensor.data()),
                                       tensor_elems * sizeof(tensor_value_type),
                                       cudaMemcpyDeviceToDevice));

        auto iter   = cute::make_gmem_ptr(thrust::raw_pointer_cast(device_vector.data()));
        auto tensor = cute::make_tensor(iter, old_tensor.layout());

        return cuda::std::make_tuple(std::move(device_vector), tensor);
    }

    using cute::conditional_return;
    using cute::raw_pointer_cast;
    using cute::size;
} // namespace tutorial
