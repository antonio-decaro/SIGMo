cmake_minimum_required(VERSION 3.14)
project(SubgraphIsoTests)

# Enable testing
enable_testing()

# List of source files
set(TEST_FILES
  read_write.cpp
  graphs.cpp
  candidates.cpp
  filter.cpp
)

# Add GoogleTest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/release-1.11.0.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Define the test data path
set(TEST_DATA_PATH "${CMAKE_SOURCE_DIR}/test/data/test_data.dat")
set(TEST_QUERY_PATH "${CMAKE_SOURCE_DIR}/test/data/test_query.dat")
set(TEST_POOL_PATH "${CMAKE_SOURCE_DIR}/test/data/test_pool.bin")
set(TEST_TMP_PATH "${CMAKE_SOURCE_DIR}/test/data/tmp")

# Add SYCL flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=${MBSM_TARGET_ARCHITECTURE}")

# Loop to add executables and include directories
foreach(TEST_FILE ${TEST_FILES})
  get_filename_component(EXECUTABLE_NAME ${TEST_FILE} NAME_WE)
  add_executable(test_${EXECUTABLE_NAME} ${TEST_FILE})
  target_include_directories(test_${EXECUTABLE_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_compile_definitions(test_${EXECUTABLE_NAME} PRIVATE TEST_DATA_PATH="${TEST_DATA_PATH}")
  target_compile_definitions(test_${EXECUTABLE_NAME} PRIVATE TEST_QUERY_PATH="${TEST_QUERY_PATH}")
  target_compile_definitions(test_${EXECUTABLE_NAME} PRIVATE TEST_POOL_PATH="${TEST_POOL_PATH}")
  target_compile_definitions(test_${EXECUTABLE_NAME} PRIVATE TEST_TMP_PATH="${TEST_TMP_PATH}")

  target_link_libraries(test_${EXECUTABLE_NAME} gtest gtest_main)
endforeach()

# Add tests
foreach(TEST_FILE ${TEST_FILES})
  get_filename_component(EXECUTABLE_NAME ${TEST_FILE} NAME_WE)
  add_test(NAME test_${EXECUTABLE_NAME} COMMAND test_${EXECUTABLE_NAME})
endforeach()
