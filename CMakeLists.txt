cmake_minimum_required(VERSION 3.1)
project("Seayon")
set(CMAKE_CXX_STANDARD 17)

# if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
#     add_compile_options(-fdiagnostics-color=always)
# elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
#     add_compile_options(-fcolor-diagnostics)
# endif()

option(CUDA_DISABLED ON)
set(CUDA_DISABLED ON)

# add_library(seayon STATIC "src/seayon.cpp")
# target_include_directories(seayon PUBLIC "include")

include_directories("include")
include_directories("external/timez")
add_subdirectory("examples")

message(STATUS " compiler: '${CMAKE_CXX_COMPILER_ID}'")
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CUDA_DISABLED ON)

    set(CMAKE_CXX_FLAGS_DEBUG "-g")
    set(CMAKE_CXX_FLAGS_DEBUG "-wall")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
    # set(CMAKE_CXX_FLAGS_RELEASE "-Ofast")
    # set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -ftree-vectorize -ftree-parallelize-loops=4")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2")
endif()

message(STATUS " build: '${CMAKE_BUILD_TYPE}'")
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    include(CTest)
    add_subdirectory("tests")
    add_subdirectory("external/googletest")
endif()