cmake_minimum_required(VERSION 3.21)
project(xfac_quad VERSION 0.1.0 LANGUAGES CXX)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

find_package(QTgrid-quad CONFIG REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)
set(XFAC_QUAD_EXTERN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../extern"
    CACHE PATH "Shared external sources (Catch2)")
option(BUILD_SHARED_LIBS "Build shared instead of static libraries" OFF)
option(XFAC_QUAD_BUILD_TESTS "Build xfac_quad tests" ${PROJECT_IS_TOP_LEVEL})

add_library(xfac_quad
    src/double.cpp src/complex_double.cpp src/dd128.cpp
    src/complex_dd128.cpp src/float128.cpp src/complex_float128.cpp)
add_library(xfac_quad::xfac_quad ALIAS xfac_quad)
set_target_properties(xfac_quad PROPERTIES
    POSITION_INDEPENDENT_CODE ON INSTALL_RPATH_USE_LINK_PATH TRUE)
target_compile_features(xfac_quad PUBLIC cxx_std_20)
target_include_directories(xfac_quad PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(xfac_quad PUBLIC qtgrid_quad::qtgrid_quad OpenMP::OpenMP_CXX)

if(XFAC_QUAD_BUILD_TESTS)
    enable_testing()
    find_path(CATCH2_INCLUDE_DIR catch2/catch.hpp
        PATHS "${XFAC_QUAD_EXTERN_DIR}/Catch2/single_include" NO_DEFAULT_PATH NO_CACHE REQUIRED)
    # Compile Catch2's implementation only once; it is never a production dependency.
    add_library(xfac_quad_test_main STATIC test/test_main.cpp)
    target_compile_features(xfac_quad_test_main PRIVATE cxx_std_20)
    target_include_directories(xfac_quad_test_main SYSTEM PRIVATE "${CATCH2_INCLUDE_DIR}")
    file(GLOB test_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_*.cpp")
    list(REMOVE_ITEM test_sources "${CMAKE_CURRENT_SOURCE_DIR}/test/test_main.cpp")
    foreach(source IN LISTS test_sources)
        get_filename_component(name "${source}" NAME_WE)
        set(target "xfac_quad_${name}")
        add_executable(${target} "${source}")
        target_include_directories(${target} SYSTEM PRIVATE "${CATCH2_INCLUDE_DIR}")
        target_link_libraries(${target} PRIVATE xfac_quad xfac_quad_test_main)
        target_compile_options(${target} PRIVATE
            "$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Wreorder;-Wshadow;-UNDEBUG>")
        if(name MATCHES "^test_benchmark_")
            add_test(NAME ${target} COMMAND ${target} "[.]")
            set_tests_properties(${target} PROPERTIES LABELS slow)
        else()
            add_test(NAME ${target} COMMAND ${target})
        endif()
        file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test-output/${name}")
        set_tests_properties(${target} PROPERTIES
            WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test-output/${name}")
    endforeach()
endif()

install(TARGETS xfac_quad EXPORT xfac_quadTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/xfac_quad DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
set(package_dir "${CMAKE_INSTALL_LIBDIR}/cmake/xfac_quad")
install(EXPORT xfac_quadTargets NAMESPACE xfac_quad:: DESTINATION "${package_dir}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/xfac_quadConfig.cmake" [=[
include(CMakeFindDependencyMacro)
find_dependency(QTgrid-quad CONFIG)
find_dependency(OpenMP COMPONENTS CXX)
include("${CMAKE_CURRENT_LIST_DIR}/xfac_quadTargets.cmake")
]=])
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/xfac_quadConfigVersion.cmake"
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/xfac_quadConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/xfac_quadConfigVersion.cmake" DESTINATION "${package_dir}")
