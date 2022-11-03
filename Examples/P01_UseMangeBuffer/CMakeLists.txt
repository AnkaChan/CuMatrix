cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(P01_UseManagedBuffer LANGUAGES CXX CUDA)

find_package(CuMatrix REQUIRED PATHS ${CMAKE_CURRENT_LIST_DIR}/../../cmake)
find_package(Eigen3 REQUIRED)

## Use C++11
set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CUDA_STANDARD 17)   
set (CMAKE_CUDA_ARCHITECTURES 52)

# SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  --expt-extended-lambda; --extended-lambda; )

# message(CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS})

include_directories(
	${CU_MATRIX_INCLUDE_DIR}
	${EIGEN3_INCLUDE_DIR}
)

file(GLOB SRC
    "*.h"
    "*.cpp"
	"*.c"
	"*.cu"
	)

add_executable(P01_UseManagedBuffer 
	${SRC}
)
set(CMAKE_CUDA_ARCHITECTURES 52)
target_link_libraries(P01_UseManagedBuffer ${CU_MATRIX_LIBS})
#arget_Compile_options(P01_UseManagedBuffer PUBLIC $<$COMPILE_LANGUAGE:CUDA>:--extended-lambda)
target_compile_options(P01_UseManagedBuffer PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
                       --extended-lambda
					   --default-stream per-thread
                       >)
target_include_directories(P01_UseManagedBuffer PUBLIC 
	${CU_MATRIX_INCLUDE_DIR}
	${EIGEN3_INCLUDE_DIR}
)