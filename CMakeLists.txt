cmake_minimum_required(VERSION 3.17)

project(llama_cu LANGUAGES CXX CUDA)

option(ENABLE_CUBLAS "Enable cuBLAS library support" OFF)
option(ENABLE_CUSOLVER "Enable cuSOLVER library support" OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "89")

if(MSVC)
    set(CMAKE_CUDA_FLAGS_RELEASE "-O2")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2")
else()
    set(CMAKE_CXX_FLAGS "-std=c++17 -fno-rtti -fno-exceptions")
    set(CMAKE_CUDA_FLAGS_RELEASE "-O2 -fno-rtti -fno-exceptions")
    set(CMAKE_CXX_FLAGS_RELEASE "-O2")
endif()

find_package(CUDAToolkit REQUIRED)

add_library(llama_cu_lib STATIC llama_cu.cu llama_cu.h)
set_target_properties(llama_cu_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(llama_cu_lib PRIVATE ${CUDA_INCLUDE_DIRS})

if(ENABLE_CUBLAS)
    target_link_libraries(llama_cu_lib CUDA::cublas CUDA::cublasLt)
    target_compile_definitions(llama_cu_lib PRIVATE ENABLE_CUBLAS)
endif()

add_executable(llama_cu main.cpp)
target_link_libraries(llama_cu llama_cu_lib)
add_executable(transform transform.cu)
target_link_libraries(transform llama_cu_lib)

if(ENABLE_CUBLAS)
    target_link_libraries(transform CUDA::cublas CUDA::cublasLt)
    target_compile_definitions(transform PRIVATE ENABLE_CUBLAS)
endif()

if(ENABLE_CUSOLVER)
    target_link_libraries(transform CUDA::cusolver)
    target_compile_definitions(transform PRIVATE ENABLE_CUSOLVER)
endif()