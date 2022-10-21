#include "main.cuh"
#include "CuMatrix/MatrixOps/CuMatrix.h"

void computeDetsGPU(ManagedBuffer<float>& matsbuf, ManagedBuffer<float>& detsbuf, int numMats, int numThreads){
	parallel_for_3x3_matOps KERNEL_ARGS2((numMats + numThreads - 1) / numThreads, numThreads) (matsbuf.getGPUBuffer(), numMats,
	[dets = detsbuf.getGPUBuffer()] __device__ (float* mat, int iMat) 
	{
		dets[iMat] = CuMatrix::mat3GetDeterminant(mat);
	});
}

