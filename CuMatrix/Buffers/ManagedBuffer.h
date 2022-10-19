#pragma once
#include "cuda_runtime.h"
#include "../MatrixOps/CuMatrixDefs.h"
#include "GenericBuffer.h"

////! 32-bit floating point format.
//kFLOAT = 0,

////! IEEE 16-bit floating-point format.
//kHALF = 1,

////! 8-bit integer representing a quantized floating-point value.
//kINT8 = 2,

////! Signed 32-bit integer format.
//kINT32 = 3,

////! 8-bit boolean. 0 = false, 1 = true, other values undefined.
//kBOOL = 4
struct TypeSelceter
{
	static CudaDataType selectTypes(long v) { return  CudaDataType::kINT32; }
	static CudaDataType selectTypes(float v) { return CudaDataType::kFLOAT; }
	static CudaDataType selectTypes(short v) { return CudaDataType::kHALF; }
	static CudaDataType selectTypes(unsigned char v) { return CudaDataType::kINT8; }
	static CudaDataType selectTypes(bool v) { return CudaDataType::kBOOL; }
};

template<typename T>
class ManagedBuffer
{
public:
	ManagedBuffer(size_t in_size, bool in_useCPUBuf = false, void* in_cpuBuffer = nullptr, bool in_cpuBufferOwnership = false)
		: size(in_size)
		, gpuBuffer(in_size, TypeSelceter::selectTypes(T()))
		, cpuBuffer(in_useCPUBuf ? in_size : 0, TypeSelceter::selectTypes(T()), in_cpuBuffer, (in_useCPUBuf && in_cpuBuffer != nullptr)? in_cpuBufferOwnership : false )
	{
		
	};

	void enableCPU() {

		useCPUBuf = true;
		cpuBuffer.resize(getSize());
	}

	T* getGPUBuffer() {
		return (T*)gpuBuffer.data();
	}
	T* getCPUBuffer() {
		return (T*)cpuBuffer.data();
	}

	inline void toCPU();
	inline void toGPU();
	
	size_t getSize() {
		return size;
	}
private:

	size_t size = 0;

	bool useCPUBuf = false;

	DeviceBuffer gpuBuffer;
	HostBuffer cpuBuffer;
};

template<typename T>
inline void ManagedBuffer<T>::toCPU()
{
	if (!useCPUBuf)
	{
		enableCPU();
	}
	CHECK(cudaMemcpy(
		cpuBuffer.data(), gpuBuffer.data(), gpuBuffer.nbBytes(), cudaMemcpyDeviceToHost));

}

template<typename T>
inline void ManagedBuffer<T>::toGPU()
{
	CHECK(cudaMemcpy(
		gpuBuffer.data(), cpuBuffer.data(), cpuBuffer.nbBytes(), cudaMemcpyHostToDevice));
}