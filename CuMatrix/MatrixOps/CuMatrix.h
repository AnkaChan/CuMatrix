#pragma once
#include "cuda_runtime.h"
#include "CuMatrixDefs.h"
#include "device_launch_parameters.h"

namespace CuMatrix
{
/*
	Assuming all the matrix to be column major; 
*/

	template <typename DType>
	GPU_CPU_INLINE_FUNC DType* vecPtr(DType* buffer, int vecPos, int stride) {
		return buffer + vecPos * stride;
	}

	template <typename DType>
	GPU_CPU_INLINE_FUNC void vec3Set(DType* v, const  DType val) {
		v[0] = val;
		v[1] = val;
		v[2] = val;
	}

	template <typename DType>
	GPU_CPU_INLINE_FUNC DType vec2CrossProduct(const  DType* v1, const  DType* v2) {
		return v1[0] * v2[1] - v1[1] * v2[0];

	}

	template <typename DType>
	GPU_CPU_INLINE_FUNC void vec3Add(const DType* v1, const DType* v2, DType* result) {
		result[0] = v1[0] + v2[0];
		result[1] = v1[1] + v2[1];
		result[2] = v1[2] + v2[2];
	}

	template <typename DType>
	GPU_CPU_INLINE_FUNC void vec3Minus(const DType* v1, const DType* v2, DType* result) {
		result[0] = v1[0] - v2[0];
		result[1] = v1[1] - v2[1];
		result[2] = v1[2] - v2[2];
	}


	template <typename DType>
	GPU_CPU_INLINE_FUNC void vec3Mul(const DType* v1, const DType a, DType* result) {
		result[0] = v1[0] * a;
		result[1] = v1[1] * a;
		result[2] = v1[2] * a;
	}

	template <typename DType>
	GPU_CPU_INLINE_FUNC DType vec3DotProduct(const DType* v1, const DType* v2) {
		return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
	}

	template <typename DType>
	GPU_CPU_INLINE_FUNC void vec3CrossProduct(const DType* v1, const DType* v2, DType* result) {
		result[0] = v1[1] * v2[2] - v1[2] * v2[1];
		result[1] = v1[2] * v2[0] - v1[0] * v2[2];
		result[2] = v1[0] * v2[1] - v1[1] * v2[0];

	}

	template <typename DType>
	GPU_CPU_INLINE_FUNC DType vec3Norm(const DType* v) {
		return sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
	}


	template <typename DType>
	GPU_CPU_INLINE_FUNC DType vec3NormSquare(const DType* v) {
		return (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
	}

	template <typename DType>
	GPU_CPU_INLINE_FUNC DType vec3TripleProduct(const DType* v1, const DType* v2, const DType* v3) {
		DType crossProduct[3];
		// AB* (AC ^ AD);
		vec3CrossProduct(v2, v3, crossProduct);

		return vec3DotProduct(v1, crossProduct);
	}

	template <typename DType>
	GPU_CPU_INLINE_FUNC DType mat3IJ(const DType* m, const int32_t row, const int32_t col) {
		return m[(3 * col) + row];
	}

	template <typename DType>
	GPU_CPU_INLINE_FUNC void mat3VecProduct(const DType* m, const DType* v, DType* result) {
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

	template <typename DType>
	GPU_CPU_INLINE_FUNC void mat3MatProduct(const DType* inA, const DType* inB, DType* outC) {
		mat3VecProduct(inA, inB, outC);
		mat3VecProduct(inA, inB+3, outC+3);
		mat3VecProduct(inA, inB+6, outC+6);
	}

	template <typename DType>
	GPU_CPU_INLINE_FUNC DType mat3Determinant(const DType* m) {
		const DType  a11 = m[0]; const DType  a12 = m[3]; const DType  a13 = m[6];
		const DType  a21 = m[1]; const DType  a22 = m[4]; const DType  a23 = m[7];
		const DType  a31 = m[2]; const DType  a32 = m[5]; const DType  a33 = m[8];
		return a11 * a22 * a33 + a12 * a23 * a31 + a13 * a21 * a32 - a13 * a22 * a31 - a12 * a21 * a33 - a11 * a23 * a32;
	}

	template <typename DType>
	GPU_CPU_INLINE_FUNC bool solve3x3(const DType* m, const DType * b, DType* out)
	{
		const DType  a11 = m[0]; const DType  a12 = m[3]; const DType  a13 = m[6];
		const DType  a21 = m[1]; const DType  a22 = m[4]; const DType  a23 = m[7];
		const DType  a31 = m[2]; const DType  a32 = m[5]; const DType  a33 = m[8];

		const DType i11 = a33 * a22 - a32 * a23;
		const DType i12 = -(a33 * a12 - a32 * a13);
		const DType i13 = a23 * a12 - a22 * a13;

		const DType det = (a11 * i11 + a21 * i12 + a31 * i13);

		if (IS_ZERO_APPROX(det))
		{
			return false;
		}

		const DType deti = 1.0 / det;

		const DType i21 = -(a33 * a21 - a31 * a23);
		const DType i22 = a33 * a11 - a31 * a13;
		const DType i23 = -(a23 * a11 - a21 * a13);

		const DType i31 = a32 * a21 - a31 * a22;
		const DType i32 = -(a32 * a11 - a31 * a12);
		const DType i33 = a22 * a11 - a21 * a12;

		out[0] = deti * (i11 * b[0] + i12 * b[1] + i13 * b[2]);
		out[1] = deti * (i21 * b[0] + i22 * b[1] + i23 * b[2]);
		out[2] = deti * (i31 * b[0] + i32 * b[1] + i33 * b[2]);

		return true;
	}

	template <typename DType>
	GPU_CPU_INLINE_FUNC bool solve3x3_psd_stable(const DType* m, const DType* b, DType* out)
	{
		const DType  a11 = m[0]; const DType  a12 = m[3]; const DType  a13 = m[6];
		const DType  a21 = m[1]; const DType  a22 = m[4]; const DType  a23 = m[7];
		const DType  a31 = m[2]; const DType  a32 = m[5]; const DType  a33 = m[8];

		const DType i11 = a33 * a22 - a32 * a23;
		const DType i12 = -(a33 * a12 - a32 * a13);
		const DType i13 = a23 * a12 - a22 * a13;

		const DType det = (a11 * i11 + a21 * i12 + a31 * i13);

		if (det < CMP_EPSILON * (abs(a11 * i11) + abs(a21 * i12) + abs(a31 * i13)))
		{
			return false;
		}

		const DType deti = 1.0 / det;

		const DType i21 = -(a33 * a21 - a31 * a23);
		const DType i22 = a33 * a11 - a31 * a13;
		const DType i23 = -(a23 * a11 - a21 * a13);

		const DType i31 = a32 * a21 - a31 * a22;
		const DType i32 = -(a32 * a11 - a31 * a12);
		const DType i33 = a22 * a11 - a21 * a12;

		out[0] = deti * (i11 * b[0] + i12 * b[1] + i13 * b[2]);
		out[1] = deti * (i21 * b[0] + i22 * b[1] + i23 * b[2]);
		out[2] = deti * (i31 * b[0] + i32 * b[1] + i33 * b[2]);

		return true;
	}
};

template <class Func, typename DType>
__global__ void parallel_for_3x3_matOps(DType* matsFlatten, int numMats, Func func) {
	for (int i = blockIdx.x * blockDim.x + threadIdx.x;
		i < numMats;
		i += blockDim.x * gridDim.x)
	{
		func(matsFlatten + 9 * i, i);
	}
}


// multiplying 2 mat with abitary dimensions
template <typename DType>
__global__ void multiplicateMatrixOnDevice(DType* array_A, DType* array_B, DType* array_C, int M_p, int K_p, int N_p)
{
	int ix = threadIdx.x + blockDim.x * blockIdx.x;//row number
	int iy = threadIdx.y + blockDim.y * blockIdx.y;//col number

	if (ix < N_p && iy < M_p)
	{
		DType sum = 0;
		for (int k = 0; k < K_p; k++)
		{
			sum += array_A[iy * K_p + k] * array_B[k * N_p + ix];
		}
		array_C[iy * N_p + ix] = sum;
	}
}


