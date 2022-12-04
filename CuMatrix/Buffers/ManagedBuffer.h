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
	static CudaDataType selectTypes(int32_t v) { return  CudaDataType::kINT32; }
	static CudaDataType selectTypes(float v) { return CudaDataType::kFLOAT; }
	static CudaDataType selectTypes(short v) { return CudaDataType::kHALF; }
	static CudaDataType selectTypes(unsigned char v) { return CudaDataType::kINT8; }
	static CudaDataType selectTypes(bool v) { return CudaDataType::kBOOL; }
};

template<typename T>
class ManagedBuffer
{
public:
	typedef std::shared_ptr<ManagedBuffer<T>> SharedPtr;
	typedef ManagedBuffer<T>* Ptr;

	ManagedBuffer(size_t in_size, bool in_useCPUBuf = false, T* in_cpuBuffer = nullptr, bool in_cpuBufferOwnership = false)
		: size(in_size)
		, gpuBuffer(in_size, TypeSelceter::selectTypes(T()))
		, cpuBuffer(in_useCPUBuf ? in_size : 0, TypeSelceter::selectTypes(T()), in_cpuBuffer, (in_useCPUBuf && in_cpuBuffer != nullptr)? in_cpuBufferOwnership : false )
	{
		if (in_cpuBuffer != nullptr)
		{
			// std::cout << "Registering address: " << in_cpuBuffer << std::endl;
			CUDA_CHECK_RET(cudaHostRegister(in_cpuBuffer, cpuBuffer.nbBytes(), cudaHostRegisterDefault));

		}
	};

	void enableCPU() {
		cpuBuffer.resize(getSize());
	}

	T* getGPUBuffer() {
		return (T*)gpuBuffer.data();
	}
	T* getCPUBuffer() {
		return (T*)cpuBuffer.data();
	}

	inline void toCPU();
	inline void copyToExternalCPUBuffer(void * pExternalCPUBuffer);
	inline void toGPU();
	
	// return the number of elements, not the memory size messured by bytes
	size_t getSize() {
		return size;
	}

	size_t nBytes() {
		return gpuBuffer.nbBytes();
	}
private:

	size_t size = 0;

	DeviceBuffer gpuBuffer;
	HostBuffer cpuBuffer;
};

template<typename T>
inline void ManagedBuffer<T>::toCPU()
{
	if (getCPUBuffer() == nullptr)
	{
		enableCPU();
	}
	CUDA_CHECK_RET(cudaMemcpy(
		cpuBuffer.data(), gpuBuffer.data(), gpuBuffer.nbBytes(), cudaMemcpyDeviceToHost));

}

template<typename T>
inline void ManagedBuffer<T>::copyToExternalCPUBuffer(void* pExternalCPUBuffer)
{
	CUDA_CHECK_RET(cudaMemcpy(
		pExternalCPUBuffer, gpuBuffer.data(), gpuBuffer.nbBytes(), cudaMemcpyDeviceToHost));
}

template<typename T>
inline void ManagedBuffer<T>::toGPU()
{
	CUDA_CHECK_RET(cudaMemcpy(
		gpuBuffer.data(), cpuBuffer.data(), cpuBuffer.nbBytes(), cudaMemcpyHostToDevice));
}