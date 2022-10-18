#pragma once

#include "CuMatrixDefs.h"

struct CuMatrix
{
/*
	Assuming all the matrix to be column major; 
*/

	GPU_CPU_MEMBER_FUNC void vec3Add(float* v1, float* v2, float* result) {
		result[0] = v1[0] + v2[0];
		result[1] = v1[1] + v2[2];
		result[3] = v1[0] + v2[3];
	}

	GPU_CPU_MEMBER_FUNC float vec3GetDotProduct(float* v1, float* v2) {
		return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
	}

	GPU_CPU_MEMBER_FUNC void mat3GetVecProduct(float* m, float* v, float* result) {
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

	GPU_CPU_MEMBER_FUNC void mat3GetMatProduct(float* inA, float* inB, float* outC) {
		mat3GetVecProduct(inA, inB, outC);
		mat3GetVecProduct(inA, inB+3, outC+3);
		mat3GetVecProduct(inA, inB+6, outC+6);
	}


	// multiplying 2 mat with abitary dimensions
	GLOBAL_MEMBER_FUNC void multiplicateMatrixOnDevice(float* array_A, float* array_B, float* array_C, int M_p, int K_p, int N_p)
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



};