#pragma once
#include "cuda_runtime.h"
#include "CuMatrixDefs.h"
#include "device_launch_parameters.h"

namespace CuMatrix
{
/*
	Assuming all the matrix to be column major; 
*/

	GPU_CPU_INLINE_FUNC float* vecPtr(float* buffer, int vecPos, int stride) {
		return buffer + vecPos * stride;
	}


	GPU_CPU_INLINE_FUNC float vec2CrossProduct(float* v1, float* v2) {
		return v1[0] * v2[1] - v1[1] * v2[0];

	}

	GPU_CPU_INLINE_FUNC void vec3Add(float* v1, float* v2, float* result) {
		result[0] = v1[0] + v2[0];
		result[1] = v1[1] + v2[1];
		result[2] = v1[2] + v2[2];
	}

	GPU_CPU_INLINE_FUNC void vec3Minus(float* v1, float* v2, float* result) {
		result[0] = v1[0] - v2[0];
		result[1] = v1[1] - v2[1];
		result[2] = v1[2] - v2[2];
	}

	GPU_CPU_INLINE_FUNC void vec3Mul(float* v1, float a, float* result) {
		result[0] = v1[0] * a;
		result[1] = v1[1] * a;
		result[2] = v1[2] * a;
	}

	GPU_CPU_INLINE_FUNC float vec3DotProduct(float* v1, float* v2) {
		return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
	}

	GPU_CPU_INLINE_FUNC void vec3CrossProduct(float* v1, float* v2, float* result) {
		result[0] = v1[1] * v2[2] - v1[2] * v2[1];
		result[1] = v1[2] * v2[0] - v1[0] * v2[2];
		result[2] = v1[0] * v2[1] - v1[1] * v2[0];

	}

	GPU_CPU_INLINE_FUNC float vec3Norm(float* v) {
		return sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
	}


	GPU_CPU_INLINE_FUNC float vec3NormSquare(float* v) {
		return (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
	}

	GPU_CPU_INLINE_FUNC float vec3TripleProduct(float* v1, float* v2, float* v3) {
		float crossProduct[3];
		// AB* (AC ^ AD);
		vec3CrossProduct(v2, v3, crossProduct);

		return vec3DotProduct(v1, crossProduct);
	}

	GPU_CPU_INLINE_FUNC float mat3IJ(float* m, int32_t row, int32_t col) {
		return m[(3 * col) + row];
	}

	GPU_CPU_INLINE_FUNC void mat3VecProduct(float* m, float* v, float* result) {
		result[0] = m[0] * v[0];
		result[1] = m[1] * v[0];
		result[2] = m[2] * v[0];

		result[0] += m[3] * v[1];
		result[1] += m[4] * v[1];
		result[2] += m[5] * v[1];

		result[0] += m[6] * v[2];
		result[1] += m[7] * v[2];
		result[2] += m[8] * v[2];

	}

	GPU_CPU_INLINE_FUNC void mat3MatProduct(float* inA, float* inB, float* outC) {
		mat3VecProduct(inA, inB, outC);
		mat3VecProduct(inA, inB+3, outC+3);
		mat3VecProduct(inA, inB+6, outC+6);
	}

	GPU_CPU_INLINE_FUNC float mat3Determinant(float* m) {
		float  a11 = m[0]; float  a12 = m[3]; float  a13 = m[6];
		float  a21 = m[1]; float  a22 = m[4]; float  a23 = m[7];
		float  a31 = m[2]; float  a32 = m[5]; float  a33 = m[8];
		return a11 * a22 * a33 + a12 * a23 * a31 + a13 * a21 * a32 - a13 * a22 * a31 - a12 * a21 * a33 - a11 * a23 * a32;
	}


};

template <class Func>
__global__ void parallel_for_3x3_matOps(float* matsFlatten, int numMats, Func func) {
	for (int i = blockIdx.x * blockDim.x + threadIdx.x;
		i < numMats;
		i += blockDim.x * gridDim.x)
	{
		func(matsFlatten + 9 * i, i);
	}
}

// multiplying 2 mat with abitary dimensions
template <class DType>
__global__ void multiplicateMatrixOnDevice(DType* array_A, DType* array_B, DType* array_C, int M_p, int K_p, int N_p)
{
	int ix = threadIdx.x + blockDim.x * blockIdx.x;//row number
	int iy = threadIdx.y + blockDim.y * blockIdx.y;//col number

	if (ix < N_p && iy < M_p)
	{
		float sum = 0;
		for (int k = 0; k < K_p; k++)
		{
			sum += array_A[iy * K_p + k] * array_B[k * N_p + ix];
		}
		array_C[iy * N_p + ix] = sum;
	}
}