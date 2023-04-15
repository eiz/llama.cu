cmake_minimum_required(VERSION 3.17)

project(llama_cu LANGUAGES CXX CUDA)

option(ENABLE_CUBLAS "Enable cuBLAS library support" OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "89")

if(MSVC)
    set(CMAKE_CUDA_FLAGS_RELEASE "-O2")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2")
else()
    set(CMAKE_CUDA_FLAGS_RELEASE "-O2 -fno-rtti -fno-exceptions")
    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -fno-rtti -fno-exceptions")
endif()

find_package(CUDAToolkit REQUIRED)

add_library(llama_cu_lib STATIC
    llama_cu.cu
    llama_cu.h
)

set_target_properties(llama_cu_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(llama_cu main.cu)
target_link_libraries(llama_cu llama_cu_lib)
target_include_directories(llama_cu PRIVATE ${CUDA_INCLUDE_DIRS})

if(ENABLE_CUBLAS)
    if (UNIX AND NOT APPLE)
        message(STATUS "Linking to static cuBLAS library on Linux")
        target_link_libraries(llama_cu ${CUDA_LIBRARIES} CUDA::cublas_static CUDA::cublasLt_static)
    elseif (WIN32)
        message(STATUS "Linking to dynamic cuBLAS library on Windows")
        target_link_libraries(llama_cu ${CUDA_LIBRARIES} CUDA::cublas CUDA::cublasLt)
    endif()
    target_compile_definitions(llama_cu PRIVATE ENABLE_CUBLAS)
endif()
