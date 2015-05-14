/*
 * test.cpp
 *
 *  Created on: May 28, 2014
 *      Author: Adam Kosiorek
 */


#include "gtest/gtest.h"

#include "Filter.h"
#include "FilterFactory.h"

class FilterFactoryTest : public testing::Test {};

TEST_F(FilterFactoryTest, AllowedParamtersTest) {

    struct Params {
        float t0;
        float tk;
        float timeResolution;
        int xRange;
        int yRange;
    };

    std::vector<Params> invalidParams = {
            {0, 0, 0, 0, 0},    // all zero
            {1, 1, 2, 3, 4},    // t0 == tk
            {2, 1, 1, 1, 1},    // t0 < tk
            {1, 2, 0, 3, 4},    // timeResolution 0= 0
            {1, 2, 3, 0, 4},    // xRange == 0
            {1, 2, 3, -1, 4},   // negative xRange
            {1, 2, 3, 4, 0},    // yRange == 0;
            {1, 2, 3, 4, -1}    // negative yRange
    };

    for(const auto& p : invalidParams) {
        ASSERT_THROW(FilterFactory(p.t0, p.tk, p.timeResolution, p.xRange, p.yRange), std::invalid_argument);
    }

    // valid arguments
    ASSERT_NO_THROW(FilterFactory(1, 2, 3, 4, 5));
}

// similar for another angle, at least 3 time slices, odd and even ranges
TEST_F(FilterFactoryTest, SmallFilterTest) {

    int angle = 45;
    float t0 = 0;
    float tk = 0.01;
    float timeResolution = 0.01;
    int xRadius = 4;
    int yRadius = 4;
    int xSize = 2 * xRadius + 1;
    int ySize = 2 * yRadius + 1;
    FilterFactory factory(t0, tk, timeResolution, xRadius, yRadius);


    auto filter = factory.createFilter(angle);
    ASSERT_EQ(filter->angle(), angle);
    ASSERT_EQ(filter->numSlices(), (tk - t0) /timeResolution);
    ASSERT_EQ(filter->xSize(), 2 * xRadius + 1);
    ASSERT_EQ(filter->ySize(), 2 * yRadius + 1);
    auto filterSlice = filter->at(0);

    // prepare values we expect the filter to have
    decltype(filterSlice) expectedFilter(2 * xRadius + 1, 2 * yRadius + 1);
    expectedFilter<< 0.0437, 0.0412, 0.0207, -0.0132, -0.0473, -0.0671, -0.0653, -0.0456, -0.0194
            , 0.0412, 0.0220, -0.0149, -0.0571, -0.0864, -0.0896, -0.0667, -0.0302, 0.0032
            , 0.0207, -0.0149, -0.0609, -0.0980, -0.1083, -0.0858, -0.0414, 0.0047, 0.0349
            , -0.0132, -0.0571, -0.0980, -0.1153, -0.0974, -0.0500, 0.0061, 0.0479, 0.0632
            , -0.0473, -0.0864, -0.1083, -0.0974, -0.0533, 0.0069, 0.0579, 0.0814, 0.0747
            , -0.0671, -0.0896, -0.0858, -0.0500, 0.0069, 0.0617, 0.0923, 0.0902, 0.0636
            , -0.0653, -0.0667, -0.0414, 0.0061, 0.0579, 0.0923, 0.0961, 0.0721, 0.0356
            , -0.0456, -0.0302, 0.0047, 0.0479, 0.0814, 0.0902, 0.0721, 0.0379, 0.0038
            , -0.0194, 0.0032, 0.0349, 0.0632, 0.0747, 0.0636, 0.0356, 0.0038, -0.0190;
    expectedFilter = expectedFilter * 0.001;

    for(int x = 0; x < xSize; ++x) {
        for(int y = 0; y < ySize; ++y) {
            ASSERT_FLOAT_EQ(filterSlice(x, y), expectedFilter(x, y));
        }
    }

    ASSERT_EQ(filterSlice, expectedFilter);

//    angle = 190;
//    t0 = -2;
//    tk = -1.95;
//    timeResolution = 0.05;
//    xRadius = 5;
//    yRadius = 5;
//    xSize = 2 * xRadius + 1;
//    ySize = 2 * yRadius + 1;
//    FilterFactory factory2(t0, tk, timeResolution, xRadius, yRadius);
//
//
//    filter = factory2.createFilter(angle);
//    ASSERT_EQ(filter->angle(), angle);
//    ASSERT_EQ(filter->numSlices(), (tk - t0) /timeResolution);
//    ASSERT_EQ(filter->xSize(), 2 * xRadius + 1);
//    ASSERT_EQ(filter->ySize(), 2 * yRadius + 1);
//    filterSlice = filter->at(0);
//
//    // prepare values we expect the filter to have
//    expectedFilter.clear;
//    expectedFilter.resize(2 * xRadius + 1, 2 * yRadius + 1);
//    expectedFilter<< -0.0747, 0.0327, 0.0616, 0.1580, 0.1864, 0.1150, 0.0159, 0.1235, 0.1490, 0.0981, 0.0231
//            , 0.0903, 0.0242, 0.1035, 0.2207, 0.2379, 0.1251, 0.0524, 0.1832, 0.1986, 0.1181, 0.0161
//            , 0.0996, 0.0058, 0.1541, 0.2845, 0.2800, 0.1192, 0.1033, 0.2488, 0.2447, 0.1298, 0.0017
//            , 0.0997, 0.0219, 0.2071, 0.3389, 0.3033, 0.0943, 0.1636, 0.3107, 0.2788, 0.1291, 0.0195
//            , 0.0893, 0.0553, 0.2535, 0.3733, 0.3014, 0.0522, 0.2240, 0.3577, 0.2932, 0.1147, 0.0449
//            , 0.0698, 0.0884, 0.2843, 0.3801, 0.2732, 0.0000, 0.2732, 0.3801, 0.2843, 0.0884, 0.0698
//            , 0.0449, 0.1147, 0.2932, 0.3577, 0.2240, 0.0522, 0.3014, 0.3733, 0.2535, 0.0553, 0.0893
//            , 0.0195, 0.1291, 0.2788, 0.3107, 0.1636, 0.0943, 0.3033, 0.3389, 0.2071, 0.0219, 0.0997
//            , 0.0017, 0.1298, 0.2447, 0.2488, 0.1033, 0.1192, 0.2800, 0.2845, 0.1541, 0.0058, 0.0996
//            , 0.0161, 0.1181, 0.1986, 0.1832, 0.0524, 0.1251, 0.2379, 0.2207, 0.1035, 0.0242, 0.0903
//            , 0.0231, 0.0981, 0.1490, 0.1235, 0.0159, 0.1150, 0.1864, 0.1580, 0.0616, 0.0327, 0.0747;
//    expectedFilter = expectedFilter * 10^-65;
//
//    for(int x = 0; x < xSize; ++x) {
//        for(int y = 0; y < ySize; ++y) {
//            ASSERT_FLOAT_EQ(filterSlice(x, y), expectedFilter(x, y));
//        }
//    }
//
//    ASSERT_EQ(filterSlice, expectedFilter);
//
//
//
//    angle = 30;
//    t0 = 1;
//    tk = 1.03;
//    timeResolution = 0.03;
//    xRadius = 2;
//    yRadius = 2;
//    xSize = 2 * xRadius + 1;
//    ySize = 2 * yRadius + 1;
//    FilterFactory factory3(t0, tk, timeResolution, xRadius, yRadius);


}

