project("Seayon")
cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
# set(CMAKE_CXX_FLAGS_RELEASE "-Ofast")
# set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -ftree-vectorize -ftree-parallelize-loops=4")

include_directories("include")
# include_directories("external/googletest/googletest/include")
include_directories("external/timez")

add_subdirectory("external/googletest")
add_subdirectory("SeayonDemo")
add_subdirectory("SeayonMnist")
add_subdirectory("SeayonTrading")
include(CTest)
add_subdirectory("Tests")