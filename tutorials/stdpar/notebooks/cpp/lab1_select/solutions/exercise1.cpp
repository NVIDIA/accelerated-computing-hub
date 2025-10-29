/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 University of Geneva. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <algorithm>
#include <chrono>
#include <execution>
#include <numeric>
#include <vector>
#include <iterator>
#include <iostream>
#include <random>
#include <ranges>

// Select elements from "v" using "pred" and copy them to "w".
template <class UnaryPredicate>
void select(const std::vector<int>& v, UnaryPredicate pred,
            std::vector<size_t>& index, std::vector<int>& w)
{
    // DONE: parallelize "select" using parallel "count_if" & "copy_if" algorithms:
    auto count = std::count_if(std::execution::par, v.begin(), v.end(), pred);
    w.resize(count);
    std::copy_if(std::execution::par, v.begin(), v.end(), w.begin(), pred);
}

// Initialize vector
void initialize(std::vector<int>& v);

// Benchmarks the implementation
template <typename Predicate>
void bench(std::vector<int>& v, Predicate&& predicate, std::vector<size_t>& index, std::vector<int>& w);

int main(int argc, char* argv[])
{
    // Read CLI arguments, the first argument is the name of the binary:
    if (argc != 2) {
        std::cerr << "ERROR: Missing length argument!" << std::endl;
        return 1;
    }

    // Read length of vector elements
    long long n = std::stoll(argv[1]);

    // Allocate the data vector
    auto v = std::vector<int>(n);

    initialize(v);

    auto predicate = [](int x) { return x % 3 == 0; };
    std::vector<size_t> index;
    std::vector<int> w;
    select(v, predicate, index, w);
    if (!std::all_of(w.begin(), w.end(), predicate) || w.empty()) {
        std::cerr << "ERROR! ";
        std::cout << "w[0.." << std::min(10, (int)w.size()) << "] = ";
        std::copy(w.begin(), w.begin() + std::min(10, (int)w.size()), std::ostream_iterator<int>(std::cout, " "));
        std::cout << std::endl;
        return EXIT_FAILURE;
    }
    std::cerr << "Check: OK, ";

    bench(v, predicate, index, w);

    return EXIT_SUCCESS;
}

void initialize(std::vector<int>& v)
{
    auto distribution = std::uniform_int_distribution<int> {0, 100};
    auto engine = std::mt19937 {1};
    std::generate(v.begin(), v.end(), [&distribution, &engine]{ return distribution(engine); });
}

template <typename Predicate>
void bench(std::vector<int>& v, Predicate&& predicate, std::vector<size_t>& index, std::vector<int>& w) {
    // Measure bandwidth in [GB/s]
    using clk_t = std::chrono::steady_clock;
    select(v, predicate, index, w);
    auto start = clk_t::now();
    int nit = 10;
    for (int it = 0; it < nit; ++it) {
        select(v, predicate, index, w);
    }
    auto seconds = std::chrono::duration<double>(clk_t::now() - start).count(); // Duration in [s]
    // Bandwith for a memcpy:
    auto gigabytes = 2. * sizeof(int) * (double)v.size() * 1.e-9; // GB
    std::cerr << "Problem size: " << gigabytes << " GB, Bandwidth [GB/s]: " << (gigabytes * (double)nit / seconds) << std::endl;
}
