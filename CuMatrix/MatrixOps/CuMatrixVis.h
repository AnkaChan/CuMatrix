#include "CuMatrix.h"

namespace CuMatrix
{
	template <typename DType>
	GPU_CPU_INLINE_FUNC void printMat3(const DType* mat) {
		printf("%-7f %-7f %-7f\n%-7f %-7f %-7f\n%-7f %-7f %-7f\n",
			mat[0], mat[3], mat[6],
			mat[1], mat[4], mat[7],
			mat[2], mat[5], mat[8]);
	}

	template <typename DType>
	GPU_CPU_INLINE_FUNC void printMat(const DType* mat, size_t rows, size_t cols) {
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols - 1; j++) {
				printf("%-7f ", mat[i * cols + j]);
			}
			printf("%-7f\n", mat[i * cols + cols - 1]);
		}
		printf("\n");
	}


	template <typename DType>
	GPU_CPU_INLINE_FUNC void printFloatVec(const DType* vec, size_t size) {
		for (int i = 0; i < size; i++) {
			printf("%f ", vec[i]);
		}
		printf("\n");
	}

	template <typename DType>
	GPU_CPU_INLINE_FUNC void printIntVec(const DType* vec, size_t size) {
		for (int i = 0; i < size; i++) {
			printf("%d ", vec[i]);
		}
		printf("\n");
	}

	inline __host__ __device__ void printCharVec(const int8_t* v, size_t size) {
		printf("Printing int vector of size %d\n", size);
		printf("vector address %p\n", v);
		for (int i = 0; i < size; i++) {
			printf("%d ", int(v[i]));
		}
		printf("\n");
	}
}