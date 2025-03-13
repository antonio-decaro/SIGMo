# Copyright (c) 2025 University of Salerno
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.10)

# Set the project name
project(MBSM)

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

option (MBSM_ENABLE_TEST OFF)
if (MBSM_ENABLE_TEST)
    add_subdirectory(${CMAKE_SOURCE_DIR}/test)
endif()

# Add the executable
add_executable(create_pool
    ${CMAKE_SOURCE_DIR}/src/create_pool.cpp
)
add_executable(inspector
    ${CMAKE_SOURCE_DIR}/src/inspector.cpp
)
add_executable(mbsm
    ${CMAKE_SOURCE_DIR}/src/mbsm.cpp
)

# Link cxxopts to the executables
target_link_libraries(create_pool PRIVATE cxxopts)
target_link_libraries(inspector PRIVATE cxxopts)
target_link_libraries(mbsm PRIVATE cxxopts)

# Select the target architecture
set (MBSM_TARGET_ARCHITECTURE "" CACHE STRING "Target architecture for the MBSM library")
if (MBSM_TARGET_ARCHITECTURE STREQUAL "")
    message(FATAL_ERROR "Please specify the target architecture for the MBSM library")
endif()

# Set the target architecture
set(CMAKE_CXX_FLAGS "-fsycl -fsycl-targets=${MBSM_TARGET_ARCHITECTURE}")
