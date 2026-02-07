#pragma once

#include <type_traits>
#include "../tutorial_helpers.hpp"

namespace tutorial {
    namespace detail {
        template<class T, class = void>
        struct promote;

        template<class T>
        struct promote<T, std::enable_if_t<commondx::is_signed_integral_v<T> and not is_complex<T>()>> {
            using value_type = int64_t;
        };

        template<class T>
        struct promote<T, std::enable_if_t<commondx::is_unsigned_integral_v<T> and not is_complex<T>()>> {
            using value_type = uint64_t;
        };

        template<class T>
        struct promote<T, std::enable_if_t<commondx::is_floating_point_v<T> and not is_complex<T>()>> {
            using value_type = double;
        };

        template<class T, template<class> class Complex>
        struct promote<Complex<T>, std::enable_if_t<is_complex<Complex<T>>()>> {
            using promoted_internal = typename promote<T>::value_type;

            using value_type = cublasdx::complex<promoted_internal>;
        };

        template<class ValueType>
        using get_reference_value_type_t = typename promote<ValueType>::value_type;
    } // namespace detail

    template<typename TC, typename TA = TC, typename TB = TC>
    bool is_error_acceptable(double tot_rel_err) {
        if (!std::isfinite(tot_rel_err)) {
            return false;
        }

        constexpr bool is_fp8_a_b_c =
            (commondx::is_floating_point_v<TA> and not is_complex<TA>() and sizeof(TA) == 1) ||
            (commondx::is_floating_point_v<TB> and not is_complex<TB>() and sizeof(TB) == 1) ||
            (commondx::is_floating_point_v<TC> and not is_complex<TC>() and sizeof(TC) == 1);

        constexpr bool is_fp8_a_b_c_complex =
            (commondx::is_floating_point_v<TA> and is_complex<TA>() and sizeof(TA) == 2) ||
            (commondx::is_floating_point_v<TB> and is_complex<TB>() and sizeof(TB) == 2) ||
            (commondx::is_floating_point_v<TC> and is_complex<TC>() and sizeof(TC) == 2);

        constexpr bool is_bf16_a_b_c =
            std::is_same_v<TA, __nv_bfloat16> || std::is_same_v<TB, __nv_bfloat16> || std::is_same_v<TC, __nv_bfloat16>;

        constexpr bool is_bf16_a_b_c_complex = std::is_same_v<TA, cublasdx::complex<__nv_bfloat16>> ||
                                               std::is_same_v<TB, cublasdx::complex<__nv_bfloat16>> ||
                                               std::is_same_v<TC, cublasdx::complex<__nv_bfloat16>>;

        constexpr bool is_integral =
            commondx::is_integral_v<TA> and commondx::is_integral_v<TB> and commondx::is_integral_v<TC>;

        constexpr bool is_non_float_non_double_a_b_c =
            (!std::is_same_v<TA, float> && !std::is_same_v<TA, double>) ||
            (!std::is_same_v<TB, float> && !std::is_same_v<TB, double>) ||
            (!std::is_same_v<TC, float> && !std::is_same_v<TC, double>) ||
            (!std::is_same_v<TA, cublasdx::complex<float>> && !std::is_same_v<TA, cublasdx::complex<double>>) ||
            (!std::is_same_v<TB, cublasdx::complex<float>> && !std::is_same_v<TB, cublasdx::complex<double>>) ||
            (!std::is_same_v<TC, cublasdx::complex<float>> && !std::is_same_v<TC, cublasdx::complex<double>>);

        if constexpr (is_integral) {
            if (tot_rel_err != 0.0) {
                std::cout << tot_rel_err << std::endl;
                return false;
            }
        } else if (is_fp8_a_b_c) {
            if (tot_rel_err > 7e-2) {
                std::cout << tot_rel_err << std::endl;
                return false;
            }
        } else if (is_fp8_a_b_c_complex) {
            if (tot_rel_err > 1e-1) {
                std::cout << tot_rel_err << std::endl;
                return false;
            }
        } else if (is_bf16_a_b_c_complex) {
            if (tot_rel_err > 6e-2) {
                std::cout << tot_rel_err << std::endl;
                return false;
            }
        } else if (is_bf16_a_b_c) {
            if (tot_rel_err > 5e-2) {
                std::cout << tot_rel_err << std::endl;
                return false;
            }
        } else if (is_non_float_non_double_a_b_c) {
            if (tot_rel_err > 1e-2) {
                std::cout << tot_rel_err << std::endl;
                return false;
            }
        } else { // A,B,C are either float or double
            if (tot_rel_err > 1e-3) {
                std::cout << tot_rel_err << std::endl;
                return false;
            }
        }
        return true;
    }

    template<class T>
    constexpr bool is_reference_type() {
        return std::is_same_v<T, detail::get_reference_value_type_t<T>>;
    }

    template<typename T1, typename T2>
    std::enable_if_t<not is_reference_type<T1>() or not std::is_same_v<T1, T2>, double> calculate_error(
        const std::vector<T1>& data,
        const std::vector<T2>& reference,
        bool                   verbose = false,
        bool                   print   = false) {
        using ref_t = detail::get_reference_value_type_t<T2>;
        std::vector<ref_t> input_upcasted;
        std::transform(std::cbegin(data), std::cend(data), std::back_inserter(input_upcasted), converter<ref_t> {});

        // if only the input data required conversion, run comparison
        if constexpr (is_reference_type<T2>()) {
            return calculate_error(input_upcasted, reference, verbose, print);
        }
        // else, if the reference was also calculated in lower precision,
        // also upcast it and only then run the comparison
        else {
            std::vector<ref_t> reference_upcasted;
            std::transform(std::cbegin(reference),
                           std::cend(reference),
                           std::back_inserter(reference_upcasted),
                           converter<ref_t> {});
            return calculate_error(input_upcasted, reference_upcasted, verbose, print);
        }
    }

    template<class TA, class TB, class TC, typename ResT, typename RefT>
    bool check_error_custom(const std::vector<ResT>& results,
                            const std::vector<RefT>& reference,
                            bool                     verbose = false,
                            bool                     print   = false) {
        [[maybe_unused]] constexpr bool is_floating = commondx::is_floating_point_v<RefT>;
        [[maybe_unused]] constexpr bool is_integral = commondx::is_integral_v<RefT>;

        auto ret = false;

        if constexpr (is_floating) {
            double error = calculate_error(results, reference, verbose, print);
            ret          = is_error_acceptable<TC, TA, TB>(error);
        } else if constexpr (is_integral) {
            // If the input was integral, then we want absolute equality
            if (print) {
                std::cout << "Ref\tRes\n";
            }
            ret = std::equal(reference.cbegin(), reference.cend(), results.cbegin(), [print](auto ref, auto res) {
                if (print) {
                    if constexpr (is_complex<decltype(ref)>()) {
                        std::cout << ref.real() << "," << ref.imag() << "\t" << res.real() << "," << res.imag() << "\n";
                    } else {
                        std::cout << ref << "\t" << res << "\n";
                    }
                }
                if constexpr (is_complex<decltype(ref)>()) {
                    return ref.real() == res.real() and ref.imag() == res.imag();
                } else {
                    return ref == res;
                }
            });
        } else {
            static_assert(is_floating or is_integral,
                          "Reference and result must either both be integral or floating point.");
        }

        return ret;
    }
    template<class BLAS, typename ResT, typename RefT>
    bool check_error(const std::vector<ResT>& results,
                     const std::vector<RefT>& reference,
                     bool                     verbose = false,
                     bool                     print   = false) {
        using a_prec_t = typename cublasdx::precision_of<BLAS>::a_type;
        using b_prec_t = typename cublasdx::precision_of<BLAS>::b_type;
        using c_prec_t = typename cublasdx::precision_of<BLAS>::c_type;
        return check_error_custom<a_prec_t, b_prec_t, c_prec_t, ResT, RefT>(results, reference, verbose, print);
    }

    template<typename ResT, typename RefT>
    bool check_error(const std::vector<ResT>& results,
                     const std::vector<RefT>& reference,
                     bool                     verbose = false,
                     bool                     print   = false) {
        return check_error_custom<ResT, ResT, ResT>(results, reference, verbose, print);
    }

    template<typename T>
    double calculate_error(const std::vector<T>& data, const std::vector<T>& reference, bool verbose, bool print) {
        using std::abs;
        using std::sqrt;

        // Use either double or complex<double> for error computation
        using value_type = cute::remove_cvref_t<decltype(reference[0])>;
        using error_type = std::conditional_t<is_complex<value_type>(), cublasdx::complex<double>, double>;

        if (print && verbose) {
            printf("Idx:\tVal\tRefVal\tRelError\n");
        }

        double eps = 1e-200;

        double tot_error_sq    = 0;
        double tot_norm_sq     = 0;
        double tot_res_norm_sq = 0;
        double tot_ind_rel_err = 0;
        double max_ind_rel_err = 0;
        double max_ind_abs_err = 0;
        for (std::size_t i = 0; i < data.size(); ++i) {
            error_type val = convert<error_type>(data[i]);
            error_type ref = convert<error_type>(reference[i]);

            double aref = detail::cbabs(ref);
            double aval = detail::cbabs(val);
            double diff = std::abs(aref - aval);

            double rel_error = diff / (aref + eps);

            // Individual relative error
            tot_ind_rel_err += rel_error;

            // Maximum relative error
            max_ind_rel_err = std::max(max_ind_rel_err, rel_error);
            max_ind_abs_err = std::max(max_ind_abs_err, diff);

            // Total relative error
            tot_error_sq += diff * diff;
            tot_norm_sq += aref * aref;

            const double inc = detail::cbabs(val) * detail::cbabs(val);
            tot_res_norm_sq += inc;

            if ((print && verbose) and (detail::cbabs(diff) > 0.01)) {
                if constexpr (is_complex<error_type>()) {
                    std::cout << i << ":\t" << '<' << val.real() << ',' << val.imag() << '>' << "\t" << '<'
                              << ref.real() << ',' << ref.imag() << '>' << "\t" << rel_error << "\n";
                } else {
                    std::cout << i << ":\t" << val << "\t" << ref << "\t" << rel_error << "\n";
                }
            }
        }

        if (print)
            printf("Vector reference  norm: [%.5e]\n", sqrt(tot_norm_sq));

        if (print)
            printf("Vector result  norm: [%.5e]\n", sqrt(tot_res_norm_sq));

        double tot_rel_err = sqrt(tot_error_sq / (tot_norm_sq + eps));
        if (print)
            printf("Vector  relative error: [%.5e]\n", tot_rel_err);

        double ave_rel_err = tot_ind_rel_err / double(data.size());
        if (print)
            printf("Average relative error: [%.5e]\n", ave_rel_err);

        if (print)
            printf("Maximum relative error: [%.5e]\n", max_ind_rel_err);

        if (print)
            printf("Maximum absolute error: [%.5e]\n", max_ind_abs_err);

        return tot_rel_err;
    }
} // namespace tutorial
