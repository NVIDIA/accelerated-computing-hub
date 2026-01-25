
    int const warm_up_runs = 10;
    int const kernel_runs = 100;

    // ===================================
    // Problem configuration
    // ===================================

    // (syrk_n, syrk_k, alpha, beta, uplo)
    std::vector<tutorial::syrk_problem_t> problems = {
        {2048, 2048, 0.9, 1.1, tutorial::matrix_half::upper},
        {2048, 2048, 0.9, 1.1, tutorial::matrix_half::lower}
    };
    

    // ===================================
    // Global SYRK configuration
    // ===================================

    // The number of slices used in emulation algorithm
    // More slices = higher precision but more computation
    constexpr unsigned slices = 7;

    bool const   debug        = false;
