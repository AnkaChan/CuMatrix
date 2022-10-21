#pragma once
#include "CuMatrix/Buffers/ManagedBuffer.h"
#include "CuMatrix/Interface/EigenInterface.h"

void computeDetsGPU(ManagedBuffer<float>& matsbuf, ManagedBuffer<float>& detsbuf, int numMats, int numThreads);
