//
// Created by drinkingcoder on 17-10-31.
//

#ifndef CUDA_GUASSIAN_BLUR_KERNEL_HPP
#define CUDA_GUASSIAN_BLUR_KERNEL_HPP

#include <memory>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>


class Kernel {
public:
    Kernel(){}
    virtual ~Kernel(){}

    virtual void preparation() = 0;
    virtual void awakeKernel() = 0;
    virtual void postProcessing() = 0;
};

static cudaEvent_t m_start,m_stop;
static float m_duration;

class KernelTimeTest {
public:
    KernelTimeTest(){
        cudaEventCreate(&m_start);
        cudaEventCreate(&m_stop);
    }
    virtual ~KernelTimeTest(){}
    void testSingleTime(std::shared_ptr<Kernel> kernel) {
        kernel->preparation();

        cudaEventRecord(m_start,NULL);
        kernel->awakeKernel();
        cudaEventRecord(m_stop,NULL);

        cudaEventSynchronize(m_start);
        cudaEventSynchronize(m_stop);
        cudaEventElapsedTime(&m_duration,m_start,m_stop);

        kernel->postProcessing();
    }
    void testMultipleTimes(std::shared_ptr<Kernel> kernel,size_t times) {
        float duration = 0;
        for(int i = 0; i < times; i++) {
            testSingleTime(kernel);
            duration+=m_duration;
        }
        m_duration = duration/times;
    }
    float getDuration() { return m_duration; }
protected:
private:
//    cudaEvent_t m_start,m_stop;
//    float m_duration = 0.0f;
};

#endif //CUDA_GUASSIAN_BLUR_KERNEL_HPP
