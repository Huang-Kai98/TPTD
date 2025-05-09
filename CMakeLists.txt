cmake_minimum_required(VERSION 3.10)
project(TPTD)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(TBB REQUIRED)
find_package(Threads REQUIRED)
find_package(MKL REQUIRED PATHS $ENV{MKLROOT})
set(TPTD MultipleShooting Parareal)
foreach(target ${TPTD})
    add_executable(${target} ${target}.cpp)
    target_include_directories(${target} PUBLIC ${TBB_INCLUDE_DIRS} ${MKL_INCLUDE_DIRS})
    target_link_libraries(${target} TBB::tbb Threads::Threads MKL::MKL)
endforeach()

