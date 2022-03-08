cmake_minimum_required(VERSION 3.20)
project(work_proj)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS -fopenmp)

add_executable(work_proj main.cpp Matrix.h Matrix.cpp profile.h)
