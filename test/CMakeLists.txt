cmake_minimum_required(VERSION 3.15)

# Find Catch2
find_package(Catch2 QUIET)
if(NOT Catch2_FOUND)
    # Use bundled Catch2 submodule if available
    if(EXISTS "${CMAKE_SOURCE_DIR}/extern/Catch2/single_include/catch2/catch.hpp")
        add_library(Catch2_main INTERFACE)
        target_include_directories(Catch2_main INTERFACE
            "${CMAKE_SOURCE_DIR}/extern/Catch2/single_include")
    else()
        message(FATAL_ERROR "Catch2 not found")
    endif()
endif()

# ---- Test executables ----

function(add_xfac_quad_test test_name test_src)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE xfac_quad Eigen3::Eigen)
    if(Catch2_FOUND)
        target_link_libraries(${test_name} PRIVATE Catch2::Catch2)
    else()
        target_link_libraries(${test_name} PRIVATE Catch2_main)
    endif()
    target_compile_options(${test_name} PRIVATE -Wall)
    add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

add_xfac_quad_test(xfac_quad_test_SVDDecomp test_SVDDecomp.cpp)
add_xfac_quad_test(xfac_quad_test_matrix_ci test_matrix_ci.cpp)
add_xfac_quad_test(xfac_quad_test_tensor_ci test_tensor_ci.cpp)

# Enable CTest
enable_testing()
