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

option (MBSM_ENABLE_TEST OFF)
if (MBSM_ENABLE_TEST)
    add_subdirectory(${CMAKE_SOURCE_DIR}/test)
endif()

# Add the executable
add_executable(create_pool
    ${CMAKE_SOURCE_DIR}/src/create_pool.cpp
)

# Select the target architecture
set (MBSM_TARGET_ARCHITECTURE "" CACHE STRING "Target architecture for the MBSM library")
if (MBSM_TARGET_ARCHITECTURE STREQUAL "")
    message(FATAL_ERROR "Please specify the target architecture for the MBSM library")
endif()

# Set the target architecture
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsycl -fsycl-targets=${MBSM_TARGET_ARCHITECTURE}")
