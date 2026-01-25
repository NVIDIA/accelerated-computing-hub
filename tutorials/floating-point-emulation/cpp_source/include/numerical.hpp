#pragma once

namespace tutorial {

    enum class matrix_half
    {
        lower,
        upper
    };

    namespace detail {
        template<class T>
        struct is_complex_helper {
            static constexpr bool value = false;
        };

        template<class T>
        struct is_complex_helper<cublasdx::complex<T>> {
            static constexpr bool value = true;
        };

        template<class T>
        struct is_complex_helper<std::complex<T>> {
            static constexpr bool value = true;
        };

        template<class T>
        struct is_complex_helper<cuda::std::complex<T>> {
            static constexpr bool value = true;
        };
    } // namespace detail

    template<typename T>
    CUBLASDX_HOST_DEVICE constexpr bool is_complex() {
        return detail::is_complex_helper<T>::value;
    }

    namespace detail {
        template<typename T>
        double cbabs(T v) {
            if constexpr (is_complex<T>()) {
                auto imag = std::abs(static_cast<double>(v.imag()));
                auto real = std::abs(static_cast<double>(v.real()));
                return (real + imag) / 2.0;
            } else {
                return std::abs(static_cast<double>(v));
            }
        }
    } // namespace detail

    template<typename T1, typename T2>
    __host__ __device__ __forceinline__ constexpr T1 convert(T2 v) {
        constexpr bool is_output_complex = cublasdx::detail::has_complex_interface_v<T1>;
        constexpr bool is_input_complex  = cublasdx::detail::has_complex_interface_v<T2>;
        if constexpr (is_input_complex and is_output_complex) {
            using t1_vt = typename T1::value_type;
            return T1(convert<t1_vt>(v.real()), convert<t1_vt>(v.imag()));
        } else if constexpr (is_output_complex) {
            using t1_vt = typename T1::value_type;
            return T1(convert<t1_vt>(v), convert<t1_vt>(v));
        } else if constexpr (is_input_complex) {
            return convert<T1>(v.real());
        } else if constexpr (COMMONDX_STL_NAMESPACE::is_convertible_v<T2, T1>) {
            return static_cast<T1>(v);
        } else if constexpr (COMMONDX_STL_NAMESPACE::is_constructible_v<T1, T2>) {
            return T1(v);
        } else {
            static_assert(COMMONDX_STL_NAMESPACE::is_convertible_v<T2, T1>,
                          "Please provide your own conversion function");
        }
    }

    template<typename T>
    struct converter {
        template<class V>
        CUBLASDX_HOST_DEVICE constexpr T operator()(V const& v) const {
            return convert<T>(v);
        }
    };

} // namespace tutorial
