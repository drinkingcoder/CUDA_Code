/**
  * Maximum number of thread per thread: 1024
  * Warp size: 32
  */


#include <memory>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

#include "kernelGaussian.hpp"
#include "kernelGaussianSharedMemory.hpp"

using namespace cv;
using namespace std;

int main()
{
    KernelTimeTest test;
    std::shared_ptr<Kernel> kernel = make_shared<KernelGaussianSharedMemory>();
    test.testSingleTime(kernel);
//    test.testMultipleTimes(kernel,100);
    std::cout << " duration = " << test.getDuration() << std::endl;
}
