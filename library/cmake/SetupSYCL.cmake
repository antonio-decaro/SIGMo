include(CheckCXXCompilerFlag)

# Check if the compiler supports the -fsycl flag
check_cxx_compiler_flag("-fsycl" COMPILER_SUPPORTS_SYCL)

if (COMPILER_SUPPORTS_SYCL)
    message(STATUS "Compiler supports SYCL")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
else()
    message(FATAL_ERROR "Compiler does not support SYCL")
endif()
