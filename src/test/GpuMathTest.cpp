//
// Created by Adam Kosiorek on 6/12/15.
//

#include "gtest/gtest.h"
#include <thrust/complex.h>
#include <complex>
#include "common.h"
#include "TestUtils.h"
#include "GpuMath.h"
#include "DeviceBlob.h"

class RealGpuMathTest : public testing::Test {
public:
    RealGpuMathTest() :
    b1(rows, cols),
    b2(rows, cols),
    b3(rows, cols),
    tolerance(1e-8) {}

    int rows = 2;
    int cols = 3;
    DeviceBlob<float> b1;
    DeviceBlob<float> b2;
    DeviceBlob<float> b3;

    float tolerance;
};

TEST_F(RealGpuMathTest, scaleTest) {


    float x1[] = {1, 2, 3, 4, 5, 6};
    float result[] = {0.5, 1, 1.5, 2, 2.5, 3};
    this->b1.copyFrom(x1);
    gpuScale(b1.count(), 0.5, b1.data());
    b1.copyTo(x1);

    ASSERT_NEAR_VEC(x1, result, b1.count());
}

TEST_F(RealGpuMathTest, axpyTest) {


    float x1[] = {1, 2, 3, 4, 5, 6};
    float x2[] = {2, 3, 4, 5, 6, 7};
    float result[] = {2.5, 4, 5.5, 7, 8.5, 10};
    this->b1.copyFrom(x1);
    this->b2.copyFrom(x2);

    gpuAXPY(this->b1.count(), 0.5, b1.data(), b2.data());
    b2.copyTo(x1);
    ASSERT_NEAR_VEC(x1, result, this->b1.count());
}

TEST_F(RealGpuMathTest, mulToTest) {


    float x1[] = {1, 2, 3, 4, 5, 6};
    float x2[] = {2, 3, 4, 5, 6, 7};
    float x3[] = {3, 4, 5, 6, 7, 8};
    float result[] = {5, 10, 17, 26, 37, 50};
    this->b1.copyFrom(x1);
    this->b2.copyFrom(x2);
    this->b3.copyFrom(x3);

    gpuMulTo(this->b1.count(), b1.data(), b2.data(), b3.data());
    b3.copyTo(x1);
    ASSERT_NEAR_VEC(x1, result, this->b1.count());
}



class ComplexGpuMathTest : public testing::Test {
public:
    using TC = thrust::complex<float>;
    using SC = std::complex<float>;

    ComplexGpuMathTest() :
            b1(rows, cols),
            b2(rows, cols),
            b3(rows, cols),
            tolerance(1e-8) {}

    int rows = 2;
    int cols = 3;
    DeviceBlob<TC> b1;
    DeviceBlob<TC> b2;
    DeviceBlob<TC> b3;

    float tolerance;
};

TEST_F(ComplexGpuMathTest, scaleTest) {


    TC x1[] = {1, 2, 3, TC(4, 6), 5, 6};
    TC result[] = {0.5, 1, 1.5, TC(2, 3), 2.5, 3};
    this->b1.copyFrom(x1);
    gpuScale(b1.count(), 0.5, b1.data());
    b1.copyTo(x1);

    ASSERT_NEAR_VEC_COMPLEX(x1, result, b1.count());
}

TEST_F(ComplexGpuMathTest, axpyTest) {


    TC x1[] = {TC(1, 1), TC(2, 2), TC(3, 3), TC(4, 4), TC(5, 5), TC(6, 6)};
    TC x2[] = {TC(2, 2), TC(3, 3), TC(4, 4), TC(5, 5), TC(6, 6), TC(7, 7)};
    TC result[] = {TC(2.5, 2.5), TC(4, 4), TC(5.5, 5.5), TC(7, 7), TC(8.5, 8.5), TC(10, 10)};
    this->b1.copyFrom(x1);
    this->b2.copyFrom(x2);

    gpuAXPY(this->b1.count(), 0.5, b1.data(), b2.data());
    b2.copyTo(x1);
    ASSERT_NEAR_VEC_COMPLEX(x1, result, this->b1.count());
}

TEST_F(ComplexGpuMathTest, mulToTest) {


    TC x1[] = {1, TC(2, 1), 3, 4, TC(5, -1), 6};
    TC x2[] = {2, TC(3, 2), 4, 5, TC(6, 2), 7};
    TC x3[] = {3, 4, 5, 6, 7, 8};
    TC result[] = {5, TC(8, 7), 17, 26, TC(39, 4), 50};
    this->b1.copyFrom(x1);
    this->b2.copyFrom(x2);
    this->b3.copyFrom(x3);

    gpuMulTo(this->b1.count(), b1.data(), b2.data(), b3.data());
    b3.copyTo(x1);
    ASSERT_NEAR_VEC_COMPLEX(x1, result, this->b1.count());
}

TEST_F(ComplexGpuMathTest, stdToThrustTest) {
    SC x1[] = {1, SC(2, 1), 3, 4, SC(5, -1), 6};
    SC x2[] = {2, SC(3, 2), 4, 5, SC(6, 2), 7};
    SC x3[] = {3, 4, 5, 6, 7, 8};
    SC result[] = {5, SC(8, 7), 17, 26, SC(39, 4), 50};
    this->b1.copyFrom((TC*)x1);
    this->b2.copyFrom((TC*)x2);
    this->b3.copyFrom((TC*)x3);

    gpuMulTo(this->b1.count(), b1.data(), b2.data(), b3.data());
    b3.copyTo((TC*)x1);
    ASSERT_NEAR_VEC_COMPLEX(x1, result, this->b1.count());
}