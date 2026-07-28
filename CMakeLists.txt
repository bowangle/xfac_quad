cmake_minimum_required(VERSION 3.16)
project(TCIcpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---- dependencies -------------------------------------------------------------
find_package(OpenMP REQUIRED)
find_package(Boost)

add_subdirectory(extern/QTgrid-quad)

# ---- own paths ----------------------------------------------------------------
set(PROJECT_INCLUDE_DIR     ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(CATCH2_INCLUDE_DIR      ${CMAKE_CURRENT_SOURCE_DIR}/extern/Catch2/single_include)

# ---- interface target: carries our headers + transitive deps ------------------
add_library(tci_deps INTERFACE)
target_include_directories(tci_deps INTERFACE
    ${PROJECT_INCLUDE_DIR}
    ${CATCH2_INCLUDE_DIR}
)
target_link_libraries(tci_deps INTERFACE qtgrid_deps OpenMP::OpenMP_CXX Boost::boost)

# ---- test targets -------------------------------------------------------------
add_executable(test_main            test/test_main.cpp)
add_executable(Test_matrix_tci       test/test_matrix_tci.cpp)
add_executable(Test_SVDDecomp       test/test_SVDDecomp.cpp)
add_executable(Test_tensor_tci       test/test_tensor_tci.cpp)
add_executable(Test_benchmark_double   test/test_benchmark_problem_double.cpp)
add_executable(Test_benchmark_float128 test/test_benchmark_problem_float128.cpp)
add_executable(Test_benchmark_dd128    test/test_benchmark_problem_dd128.cpp)

foreach(_tgt test_main Test_matrix_tci Test_SVDDecomp Test_tensor_tci
             Test_benchmark_double Test_benchmark_float128 Test_benchmark_dd128)
    target_link_libraries(${_tgt} PRIVATE tci_deps)
    target_compile_options(${_tgt} PRIVATE -Wall -Wextra -Wreorder -Wshadow)
endforeach()
