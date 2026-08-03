cmake_minimum_required(VERSION 3.21)
project(xfac_quad LANGUAGES CXX)

if(PROJECT_IS_TOP_LEVEL)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

option(XFAC_QUAD_BUILD_TESTS
       "Build xfac_quad's own tests"
       ${PROJECT_IS_TOP_LEVEL})

# ---- dependencies -------------------------------------------------------------
find_package(OpenMP REQUIRED)
find_package(Boost REQUIRED)

# Guard against a sibling project (e.g. tci_quad) having already added its own
# copy of QTgrid-quad to this build tree: target names are global, so a second
# add_subdirectory() would be a hard "target already exists" error.
if(NOT TARGET qtgrid_deps)
    add_subdirectory(extern/QTgrid-quad)
endif()

# ---- own paths ----------------------------------------------------------------
set(XFAC_QUAD_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(CATCH2_INCLUDE_DIR    ${CMAKE_CURRENT_SOURCE_DIR}/extern/Catch2/single_include)

# ---- interface target: production headers + transitive deps -------------------
add_library(xfac_quad_deps INTERFACE)

target_include_directories(xfac_quad_deps INTERFACE
    ${XFAC_QUAD_INCLUDE_DIR})

target_compile_features(xfac_quad_deps INTERFACE cxx_std_20)

target_link_libraries(xfac_quad_deps INTERFACE
    qtgrid_deps
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

    # One entry per test/test_<name>.cpp -> target xfac_quad_test_<name>
    set(_xfac_quad_tests
        main
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
        add_executable(xfac_quad_test_${_name} test/test_${_name}.cpp)
        target_link_libraries(xfac_quad_test_${_name}
            PRIVATE xfac_quad_deps xfac_quad_testing)
        target_compile_options(xfac_quad_test_${_name} PRIVATE
            -Wall -Wextra -Wreorder -Wshadow)
        add_test(NAME xfac_quad_test_${_name}
                 COMMAND xfac_quad_test_${_name})
    endforeach()
endif()