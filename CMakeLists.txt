cmake_minimum_required(VERSION 3.21)
project(xfac_quad LANGUAGES CXX)

if(PROJECT_IS_TOP_LEVEL)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

option(XFAC_QUAD_BUILD_TESTS
       "Build QTgrid-quad, xfac_quad, and xfac_quad_runner tests"
       ${PROJECT_IS_TOP_LEVEL})

# ---- dependencies -------------------------------------------------------------
find_package(OpenMP REQUIRED)
find_package(Boost REQUIRED)

# QTGrid headers now live in this project. Only the numerical types remain
# a separate dependency; reuse its target if a sibling already added it.
if(NOT TARGET numeric_type_deps)
    add_subdirectory(extern/numeric-type-quad)
endif()

# ---- own paths ----------------------------------------------------------------
set(XFAC_QUAD_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(CATCH2_INCLUDE_DIR    ${CMAKE_CURRENT_SOURCE_DIR}/extern/Catch2/single_include)
set(SPDLOG_INCLUDE_DIR    ${CMAKE_CURRENT_SOURCE_DIR}/extern/spdlog/include)
set(JSON_INCLUDE_DIR      ${CMAKE_CURRENT_SOURCE_DIR}/extern/json/include)

if(NOT EXISTS ${JSON_INCLUDE_DIR}/nlohmann/json.hpp)
    message(FATAL_ERROR
        "nlohmann/json not found at ${JSON_INCLUDE_DIR}.\n"
        "Run: git submodule update --init --recursive")
endif()

if(NOT EXISTS ${SPDLOG_INCLUDE_DIR}/spdlog/spdlog.h)
    message(FATAL_ERROR
        "spdlog not found at ${SPDLOG_INCLUDE_DIR}.\n"
        "Run: git submodule update --init --recursive")
endif()

# ---- interface target: production headers + transitive deps -------------------
add_library(xfac_quad_deps INTERFACE)

target_include_directories(xfac_quad_deps INTERFACE
    ${XFAC_QUAD_INCLUDE_DIR})

# Header-only dependencies shared by grids, core, and runner.
target_include_directories(xfac_quad_deps SYSTEM INTERFACE
    ${SPDLOG_INCLUDE_DIR}
    ${JSON_INCLUDE_DIR})

target_compile_features(xfac_quad_deps INTERFACE cxx_std_20)

target_link_libraries(xfac_quad_deps INTERFACE
    numeric_type_deps
    OpenMP::OpenMP_CXX)

if(TARGET Boost::headers)
    target_link_libraries(xfac_quad_deps INTERFACE Boost::headers)
else()
    target_link_libraries(xfac_quad_deps INTERFACE Boost::boost)
endif()

# ---- test-only interface: Catch2 ----------------------------------------------
# Deliberately NOT part of xfac_quad_deps, so consumers of this layer never get
# Catch2 on their include path. Link this only from test targets.
add_library(xfac_quad_testing INTERFACE)

target_include_directories(xfac_quad_testing SYSTEM INTERFACE
    ${CATCH2_INCLUDE_DIR})

# ---- tests (leaf targets: nothing may ever depend on these) -------------------
if(XFAC_QUAD_BUILD_TESTS)
    if(NOT EXISTS ${CATCH2_INCLUDE_DIR}/catch2/catch.hpp)
        message(FATAL_ERROR
            "Catch2 not found at ${CATCH2_INCLUDE_DIR}.\n"
            "Run: git submodule update --init --recursive")
    endif()

    enable_testing()

    # Quantics grids: test/QTgrid-quad/test_<name>.cpp.
    foreach(_name grid multigrid)
        add_executable(qtgrid_test_${_name} test/QTgrid-quad/test_${_name}.cpp)
        target_link_libraries(qtgrid_test_${_name} PRIVATE xfac_quad_deps)
        target_compile_options(qtgrid_test_${_name} PRIVATE
            -Wall -Wextra -Wreorder -Wshadow -UNDEBUG)
        add_test(NAME qtgrid_test_${_name}
                 COMMAND qtgrid_test_${_name})
    endforeach()

    # Numerical core: test/xfac_quad/test_<name>.cpp.
    set(_xfac_quad_tests
        main
        tree
        tensor_tree
        tensor_tree_ci
        tensor_tree_ci_process
        tensor_tree_ci_parallel
        tensor_tree_ci_compare_double
        tensor_tree_ci_compare_float128
        tensor_tree_ci_compare_dd128
        tensor_tree_io
        tensor_tree_sum
        cubemat_io
        SVDDecomp_double
        SVDDecomp_float128
        SVDDecomp_dd128
        benchmark_problem_double
        benchmark_problem_float128
        benchmark_problem_dd128
        matrix_ci_double
        matrix_ci_float128
        matrix_ci_dd128
        tensor_ci_double
        tensor_ci_float128
        tensor_ci_dd128)

    foreach(_name IN LISTS _xfac_quad_tests)
        add_executable(xfac_quad_test_${_name} test/xfac_quad/test_${_name}.cpp)
        target_link_libraries(xfac_quad_test_${_name}
            PRIVATE xfac_quad_deps xfac_quad_testing)
        target_compile_options(xfac_quad_test_${_name} PRIVATE
            -Wall -Wextra -Wreorder -Wshadow)
        add_test(NAME xfac_quad_test_${_name}
                 COMMAND xfac_quad_test_${_name})
    endforeach()

    # High-level runner: test/xfac_quad_runner/test_<name>.cpp.
    set(_xfac_quad_runner_tests runner ttci_runner ttci_runner_2d determinism param)
    foreach(_name IN LISTS _xfac_quad_runner_tests)
        add_executable(xfac_quad_runner_test_${_name}
            test/xfac_quad_runner/test_${_name}.cpp)
        target_link_libraries(xfac_quad_runner_test_${_name} PRIVATE xfac_quad_deps)
        target_compile_options(xfac_quad_runner_test_${_name} PRIVATE
            -Wall -Wextra -Wreorder -Wshadow -UNDEBUG)
        add_test(NAME xfac_quad_runner_test_${_name}
                 COMMAND xfac_quad_runner_test_${_name})
    endforeach()
endif()