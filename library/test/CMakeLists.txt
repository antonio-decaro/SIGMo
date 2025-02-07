# Define the test executable
add_executable(read_graph read_graph.cpp)

# Link the test executable with the library
link_libraries(mbsm)

# Enable testing
enable_testing()

# Add tests
add_test(NAME UtilsTest COMMAND tests)
add_test(NAME IOTest COMMAND tests)
