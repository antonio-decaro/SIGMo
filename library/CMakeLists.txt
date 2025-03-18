# Copyright (c) 2025 University of Salerno
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.10)

# Set the project name
project(SIGMO)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include the SYCL setup module
include(${CMAKE_SOURCE_DIR}/cmake/SetupSYCL.cmake)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# FetchContent to download and include cxxopts
include(FetchContent)
FetchContent_Declare(
  cxxopts
  GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
  GIT_TAG v2.2.1
)
FetchContent_MakeAvailable(cxxopts)

# Include cxxopts
include_directories(${cxxopts_SOURCE_DIR}/include)

option (SIGMO_ENABLE_TEST OFF)
if (SIGMO_ENABLE_TEST)
    add_subdirectory(${CMAKE_SOURCE_DIR}/test)
endif()

# Add the executable
add_executable(create_pool
    ${CMAKE_SOURCE_DIR}/src/create_pool.cpp
)
add_executable(inspector
    ${CMAKE_SOURCE_DIR}/src/inspector.cpp
)
add_executable(sigmo
    ${CMAKE_SOURCE_DIR}/src/sigmo.cpp
)

# Link cxxopts to the executables
target_link_libraries(create_pool PRIVATE cxxopts)
target_link_libraries(inspector PRIVATE cxxopts)
target_link_libraries(sigmo PRIVATE cxxopts)

# Select the target architecture
set (SIGMO_TARGET_ARCHITECTURE "" CACHE STRING "Target architecture for the SIGMO library")
if (SIGMO_TARGET_ARCHITECTURE STREQUAL "")
    message(FATAL_ERROR "Please specify the target architecture for the SIGMO library")
endif()

# Set the target architecture
set(CMAKE_CXX_FLAGS "-fsycl -fsycl-targets=${SIGMO_TARGET_ARCHITECTURE}")
