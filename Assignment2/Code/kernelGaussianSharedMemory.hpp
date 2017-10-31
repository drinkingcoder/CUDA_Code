//
// Created by drinkingcoder on 17-10-31.
//

#ifndef CUDA_GUASSIAN_BLUR_KERNELGAUSSIANSHAREDMEMEORY_HPP
#define CUDA_GUASSIAN_BLUR_KERNELGAUSSIANSHAREDMEMEORY_HPP

#include "kernelGaussian.hpp"

class KernelGaussianSharedMemory:public KernelGaussian {
public:
    KernelGaussianSharedMemory():KernelGaussian() {}
    virtual ~KernelGaussianSharedMemory() {}

    virtual void awakeKernel();
};


#endif //CUDA_GUASSIAN_BLUR_KERNELGAUSSIANSHAREDMEMEORY_HPP
