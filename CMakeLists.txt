cmake_minimum_required(VERSION 3.17)

project(llama_cu LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS_RELEASE "-O2 -fno-rtti -fno-exceptions")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -fno-rtti -fno-exceptions")

find_package(CUDAToolkit REQUIRED)

add_executable(llama_cu main.cu)

set_target_properties(llama_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89"
)

#target_compile_options(llama_cu PRIVATE -g)
target_include_directories(llama_cu PRIVATE ${CUDA_INCLUDE_DIRS})
# CUDA::cublas_static CUDA::cublasLt_static
target_link_libraries(llama_cu ${CUDA_LIBRARIES} CUDA::cublas_static CUDA::cublasLt_static)