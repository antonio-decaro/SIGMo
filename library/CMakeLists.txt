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

# Define the GRAPH_LOCATION parameter
set(SIGMO_ALLOCATION "device" CACHE STRING "Location of the graph: host, device, or shared")
set_property(CACHE GRAPH_LOCATION PROPERTY STRINGS "host" "device" "shared")

# Map GRAPH_LOCATION to integers in compile definitions
if (SIGMO_ALLOCATION STREQUAL "host")
    message(WARNING "GRAPH_LOCATION=host is not recommended for performance. Use at your own risk.")
    add_compile_definitions(SIGMO_ALLOCATION=0)
elseif (SIGMO_ALLOCATION STREQUAL "device")
    add_compile_definitions(SIGMO_ALLOCATION=1)
elseif (SIGMO_ALLOCATION STREQUAL "shared")
    add_compile_definitions(SIGMO_ALLOCATION=2)
else()
    message(FATAL_ERROR "Invalid SIGMO_ALLOCATION value. Must be 'host', 'device', or 'shared'.")
endif()

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
