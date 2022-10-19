#include "cuda_runtime.h"
#include "../../CuMatrix/MatrixOps/CuMatrix.h"
#include "../../CuMatrix/Buffers/ManagedBuffer.h"
#include "../../CuMatrix/Interface/EigenInterface.h"
#include <Eigen/Dense>

void init3x3Mats(Eigen::VectorXf & matFlatten, size_t numMats) {

	for (size_t i = 0; i < numMats; i++)
	{
		Eigen::Map<Eigen::Matrix3f> mat(matFlatten.data() + i * 9);
		mat = Eigen::Matrix3f::Random();
	}
}

int main() {
	size_t numMats = 1000;
	size_t matW = 3;

	Eigen::VectorXf matFlatten(numMats * matW * matW);

	init3x3Mats(matFlatten, numMats);

	ManagedBuffer<float> matsbuf(matFlatten.size(), true, (void *)matFlatten.data());
	matsbuf.toGPU();

	ManagedBuffer<float> detsbuf(numMats, true, (void*)matFlatten.data());

	const int numThreads = 128;
	parallel_for_3x3_matOps KERNEL_ARGS2((numMats + numThreads - 1) / numThreads, numThreads) (matsbuf.getGPUBuffer(), numMats,
		[dets = detsbuf.getGPUBuffer()] __device__ (float* mat, int iMat) 
	{
			dets[iMat] = CuMatrix::mat3GetDeterminant(mat);
	});

	detsbuf.toCPU();

	for (size_t i = 0; i <  numMats; i++)
	{
		Eigen::Map<Eigen::Matrix3f> mat(matFlatten.data() + i * 9);
		//std::cout << "Mat: " << mat << "\n";
		std::cout << "GPU: " << detsbuf.getCPUBuffer()[i] << " | CPU:" << mat.determinant() << "\n";
	}
}