#pragma once
#include "cuda_runtime.h"
#include "CuMatrixDefs.h"
#include "device_launch_parameters.h"
#define SQR(x) ((x) * (x))
#define CUBE(x) ((x) * (x) * (x))

#ifndef __CUDACC__
#include <assert.h>
#else
#include <cassert>
#endif

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
	GPU_CPU_INLINE_FUNC void vec3Set(DType* v, const  DType val1, const  DType val2, const  DType val3) {
		v[0] = val1;
		v[1] = val2;
		v[2] = val3;
	}

	template <typename DType>
	GPU_CPU_INLINE_FUNC void vec3Set(DType* out, const DType* in) {
		out[0] = in[0];
		out[1] = in[1];
		out[2] = in[2];
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
	GPU_CPU_INLINE_FUNC void vec3MulAddTo(const DType* v1, const DType a, DType* result) {
		result[0] += v1[0] * a;
		result[1] += v1[1] * a;
		result[2] += v1[2] * a;
	}

	// result = a + l * v
	template <typename DType>
	GPU_CPU_INLINE_FUNC void vec3lerp(const DType* a, const DType l, const DType* v, DType* result) {
		result[0] = a[0] + l * v[0];
		result[1] = a[1] + l * v[1];
		result[2] = a[2] + l * v[2];
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
	GPU_CPU_INLINE_FUNC void vec3Normalize(DType* v) {
		const DType norm = vec3Norm(v);

		v[0] /= norm;
		v[1] /= norm;
		v[2] /= norm;
	}

	// aka mixed product
	template <typename DType>
	GPU_CPU_INLINE_FUNC DType vec3TripleProduct(const DType* v1, const DType* v2, const DType* v3) {
		DType crossProduct[3];
		// AB* (AC ^ AD);
		vec3CrossProduct(v2, v3, crossProduct);

		return vec3DotProduct(v1, crossProduct);
	}


	template <typename DType>
	GPU_CPU_INLINE_FUNC void vec3OuterProduct(const DType* v1, const DType* v2, DType * mat) {
		for (int iCol = 0; iCol < 3; iCol++)
		{
			for (int iRow = 0; iRow < 3; iRow++) {
				mat[iRow + 3 * iCol] = v1[iRow] * v2[iCol];
			}
		}
	}

	/*
	* Vec2
	*/
	template <typename DType>
	GPU_CPU_INLINE_FUNC DType vec2CrossProduct(const  DType* v1, const  DType* v2) {
		return v1[0] * v2[1] - v1[1] * v2[0];

	}

	template <typename DType>
	GPU_CPU_INLINE_FUNC DType vec2Norm(const  DType* v) {
		return sqrtf(v[0] * v[0] + v[1] * v[1]);;

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
	GPU_CPU_INLINE_FUNC DType mat3FNormSquare(const DType* m) {
		const DType  a11 = m[0]; const DType  a12 = m[3]; const DType  a13 = m[6];
		const DType  a21 = m[1]; const DType  a22 = m[4]; const DType  a23 = m[7];
		const DType  a31 = m[2]; const DType  a32 = m[5]; const DType  a33 = m[8];
		return a11 * a11 + a12 * a12 + a13 * a13
			+ a21 * a21 + a22 * a22 + a23 * a23
			+ a31 * a31 + a32 * a32 + a33 * a33;
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

		if (abs(det) < CMP_EPSILON * (abs(a11 * i11) + abs(a21 * i12) + abs(a31 * i13)))
		{
			out[0] = b[0];
			out[1] = b[1];
			out[2] = b[2];
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

		//DType inv[9];
		////  ( 4 8 - 5 7    5 6 - 3 8    3 7 - 4 6 ) 
		////  ( 2 7 - 1 8    0 8 - 2 6    1 6 - 0 7 )  / det
		////  ( 1 5 - 2 4    2 3 - 0 5    0 4 - 1 3 ) 

		//inv[0] = (m[4] * m[8] - m[5] * m[7]);
		//inv[3] = (m[5] * m[6] - m[3] * m[8]);
		//inv[6] = (m[3] * m[7] - m[4] * m[6]);

		//DType det = m[0] * inv[0] + m[1] * inv[3] + m[2] * inv[6];
		//if (det < CMP_EPSILON * (abs(inv[0]) + abs(inv[3]) + abs(inv[6])))
		//{
		//	out[0] = b[0];
		//	out[1] = b[1];
		//	out[2] = b[2];
		//	return false;
		//}

		//inv[1] = (m[2] * m[7] - m[1] * m[8]);
		//inv[2] = (m[1] * m[5] - m[2] * m[4]);
		//
		//inv[4] = (m[0] * m[8] - m[2] * m[6]);
		//inv[5] = (m[2] * m[3] - m[0] * m[5]);
	
		//inv[7] = (m[1] * m[6] - m[0] * m[7]);
		//inv[8] = (m[0] * m[4] - m[1] * m[3]);

		//CuMatrix:mat3VecProduct(inv, b, out);
		//CuMatrix::vec3Mul(out, 1.f / det, out);



		return true;
	}

	template <typename DType>
	struct Mat9x9Abstract {
		// column major
		DType* data;

		GPU_CPU_INLINE_FUNC Mat9x9Abstract(DType* data_in) : data(data_in) {};

		GPU_CPU_INLINE_FUNC DType* col(int iCol) { return data + iCol * 9; }
		GPU_CPU_INLINE_FUNC DType& operator() (int iRow, int iCol) { return data[iCol * 9 + iRow]; }
		GPU_CPU_INLINE_FUNC const DType& operator() (int iRow, int iCol) const { return data[iCol * 9 + iRow]; }

		GPU_CPU_INLINE_FUNC void multiplyBy(const DType mul) {
			for (size_t iCol = 0; iCol < 9; iCol++)
			{
				for (size_t iRow = 0; iRow < 9; iRow++) {
					data[iCol * 9 + iRow] *= mul;
				}
			}
		}
	};


	template <typename DType>
	struct Mat9x9Static 
	{
		DType data[81];

		GPU_CPU_INLINE_FUNC DType* col(int iCol) { return data + iCol * 9; }
		GPU_CPU_INLINE_FUNC DType& operator() (int iRow, int iCol) { return data[iCol * 9 + iRow]; }
		GPU_CPU_INLINE_FUNC const DType& operator() (int iRow, int iCol) const { return data[iCol * 9 + iRow]; }

		GPU_CPU_INLINE_FUNC void multiplyBy(const DType mul) {
			for (size_t iCol = 0; iCol < 9; iCol++)
			{
				for (size_t iRow = 0; iRow < 9; iRow++) {
					data[iCol * 9 + iRow] *= mul;
				}
			}
		}

	};

	template <typename DType, template<class> class MatType>
	GPU_CPU_INLINE_FUNC void vec9OuterProduct(const DType* v1, const DType* v2, MatType<DType>& mat) {
		for (int iCol = 0; iCol < 9; iCol++)
		{
			for (int iRow = 0; iRow < 9; iRow++) {
				mat(iRow, iCol) = v1[iRow] * v2[iCol];
			}
		}
	}

	template <typename DType>
	GPU_CPU_INLINE_FUNC void vec9Mul(const DType* v1, const DType a, DType* result) {
		result[0] = v1[0] * a;
		result[1] = v1[1] * a;
		result[2] = v1[2] * a;
		result[3] = v1[3] * a;
		result[4] = v1[4] * a;
		result[5] = v1[5] * a;
		result[6] = v1[6] * a;
		result[7] = v1[7] * a;
		result[8] = v1[8] * a;
	}

	template <typename DType>
	GPU_CPU_INLINE_FUNC void vec9Add(const DType* v1, const DType* v2, DType* result) {
		result[0] = v1[0] + v2[0];
		result[1] = v1[1] + v2[1];
		result[2] = v1[2] + v2[2];
		result[3] = v1[3] + v2[3];
		result[4] = v1[4] + v2[4];
		result[5] = v1[5] + v2[5];
		result[6] = v1[6] + v2[6];
		result[7] = v1[7] + v2[7];
		result[8] = v1[8] + v2[8];
	}

	template <typename DType>
	GPU_CPU_INLINE_FUNC void vec9MulAddTo(const DType* v1, const DType a, DType* result) {
		result[0] += v1[0] * a;
		result[1] += v1[1] * a;
		result[2] += v1[2] * a;

		result[3] += v1[3] * a;
		result[4] += v1[4] * a;
		result[5] += v1[5] * a;

		result[6] += v1[6] * a;
		result[7] += v1[7] * a;
		result[8] += v1[8] * a;
	}

	// m1 : r x c, column major
	template <typename DType, int r, int c>
	GPU_CPU_INLINE_FUNC DType& accessMatElement(DType* m, int i, int j)
	{
		assert(i < r && j < c);
		return m[r * j + i];
	}

	// m1 : r x c, column major
	template <typename DType, int r, int c>
	GPU_CPU_INLINE_FUNC const DType& accessMatElement(const DType* m, int i, int j)
	{
		assert(i < r && j < c);
		return m[r * j + i];
	}

	// m1 : r1 x c1r2, m2: c1r2 x c2, result: r1 x c2
	// all are column major
	template <typename DType, int r1, int c1r2, int c2>
	GPU_CPU_INLINE_FUNC void matMulMxN(const DType* m1, const DType* m2, DType* result)
	{
		for (int r = 0; r < r1; r++)
		{
			for (int c = 0; c < c2; c++) {
				accessMatElement<DType, r1, c2>(result, r, c) = 0.f;
				for (int i = 0; i < c1r2; i++) {
					accessMatElement<DType, r1, c2>(result, r, c) +=
						accessMatElement<DType, r1, c1r2>(m1, r, i)
						* accessMatElement<DType, c1r2, c2>(m2, i, c);
				}
			}
		}
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


