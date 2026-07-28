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
add_executable(Test_matrix_ci       test/test_matrix_tci.cpp)
add_executable(Test_SVDDecomp       test/test_SVDDecomp.cpp)
add_executable(Test_tensor_ci       test/test_tensor_tci.cpp)

foreach(_tgt test_main Test_matrix_ci Test_SVDDecomp Test_tensor_ci)
    target_link_libraries(${_tgt} PRIVATE tci_deps)
    target_compile_options(${_tgt} PRIVATE -Wall -Wextra -Wreorder -Wshadow)
endforeach()
