#include "main.cuh"
#include "Timer.h"
#include <Eigen/Dense>

void init3x3Mats(Eigen::VectorXf & matFlatten, size_t numMats) {

	for (size_t i = 0; i < numMats; i++)
	{
		Eigen::Map<Eigen::Matrix3f> mat(matFlatten.data() + i * 9);
		mat = Eigen::Matrix3f::Random();
	}
}

int main() {
	size_t numMats = 100000000;
	size_t matW = 3;

	Eigen::VectorXf matFlatten(numMats * matW * matW);

	init3x3Mats(matFlatten, numMats);

	ManagedBuffer<float> matsbuf(matFlatten.size(), true, (void*)matFlatten.data());
	matsbuf.toGPU();

	ManagedBuffer<float> detsbuf(numMats, true);

	const int numThreads = 512;

	double gpuTime = 0.;
	TICK(gpuTime);
	computeDetsGPU(matsbuf, detsbuf, numMats, numThreads);
	detsbuf.toCPU();
	TOCK(gpuTime);


	double cpuTime = 0.;
	TICK(cpuTime);
	std::vector<float> detsCpu(numMats);

	for (size_t i = 0; i < numMats; i++)
	{
		Eigen::Map<Eigen::Matrix3f> mat(matFlatten.data() + i * 9);
		detsCpu[i] = mat.determinant();
		//std::cout << "Mat: " << mat << "\n";
		/*std::cout << "GPU: " << detsbuf.getCPUBuffer()[i] << " | CPU:" << mat.determinant()
			<< " | GPU func computed on CPU: " << CuMatrix::mat3GetDeterminant(mat.data()) << "\n";*/
	}
	TOCK(cpuTime);

	for (size_t i = 0; i < numMats; i++)
	{
		if (abs(detsCpu[i] - detsbuf.getCPUBuffer()[i]) > 1e-6)
		{
			std::cout << "Bug at " << i << "th mat!\n";
			std::cout << "GPU: " << detsbuf.getCPUBuffer()[i] << " | CPU:" << detsCpu[i] << "\n";

			getchar();
		}
	}
	std::cout << detsCpu.size() << "\n";
	std::cout << "GPU time: " << gpuTime << " ms | CPU time: " << cpuTime << " ms | CPU time / 16: " << cpuTime / 16 << "ms\n";

	return 0;
}