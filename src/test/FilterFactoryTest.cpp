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
    expectedFilter<< 0.0578,  0.0580,  0.0345, -0.0081, -0.0538, -0.0833, -0.0854, -0.0631, -0.0302
            ,  0.0580,  0.0368, -0.0092, -0.0651, -0.1073, -0.1171, -0.0921, -0.0469, -0.0029
            ,  0.0345, -0.0092, -0.0693, -0.1217, -0.1415, -0.1186, -0.0643, -0.0042,  0.0382
            , -0.0081, -0.0651, -0.1217, -0.1507, -0.1346, -0.0778, -0.0054,  0.0524,  0.0772
            , -0.0538, -0.1073, -0.1415, -0.1346, -0.0828, -0.0061,  0.0633,  0.0994,  0.0964
            , -0.0833, -0.1171, -0.1186, -0.0778, -0.0061,  0.0674,  0.1128,  0.1165,  0.0865
            , -0.0854, -0.0921, -0.0643, -0.0054,  0.0633,  0.1128,  0.1241,  0.0981,  0.0529
            , -0.0631, -0.0469, -0.0042,  0.0524,  0.0994,  0.1165,  0.0981,  0.0563,  0.0120
            , -0.0302, -0.0029,  0.0382,  0.0772,  0.0964,  0.0865,  0.0529,  0.0120, -0.0194;
    expectedFilter = expectedFilter * 0.001;

    for(int x = 0; x < xSize; ++x) {
        for(int y = 0; y < ySize; ++y) {
            ASSERT_FLOAT_EQ(filterSlice(x, y), expectedFilter(x, y));
        }
    }

    ASSERT_EQ(filterSlice, expectedFilter);

//    angle = 45;
//    t0 = 0;
//    tk = 0.01;
//    timeResolution = 0.01;
//    xRadius = 4;
//    yRadius = 4;
//    FilterFactory factory2(t0, tk, timeResolution, xRadius, yRadius);
//
//
//    filter = factory.createFilter(angle);
//    ASSERT_EQ(filter->angle(), angle);
//    ASSERT_EQ(filter->numSlices(), (tk - t0) /timeResolution);
//    ASSERT_EQ(filter->xSize(), 2 * xRadius + 1);
//    ASSERT_EQ(filter->ySize(), 2 * yRadius + 1);
//    filterSlice = filter->at(0);
//
//    // prepare values we expect the filter to have
//    decltype(filterSlice) expectedFilter2(xRadius, yRadius);
//    expectedFilter2<< 0.0578,  0.0580,  0.0345, -0.0081, -0.0538, -0.0833, -0.0854, -0.0631, -0.0302
//            ,  0.0580,  0.0368, -0.0092, -0.0651, -0.1073, -0.1171, -0.0921, -0.0469, -0.0029
//            ,  0.0345, -0.0092, -0.0693, -0.1217, -0.1415, -0.1186, -0.0643, -0.0042,  0.0382
//            , -0.0081, -0.0651, -0.1217, -0.1507, -0.1346, -0.0778, -0.0054,  0.0524,  0.0772
//            , -0.0538, -0.1073, -0.1415, -0.1346, -0.0828, -0.0061,  0.0633,  0.0994,  0.0964
//            , -0.0833, -0.1171, -0.1186, -0.0778, -0.0061,  0.0674,  0.1128,  0.1165,  0.0865
//            , -0.0854, -0.0921, -0.0643, -0.0054,  0.0633,  0.1128,  0.1241,  0.0981,  0.0529
//            , -0.0631, -0.0469, -0.0042,  0.0524,  0.0994,  0.1165,  0.0981,  0.0563,  0.0120
//            , -0.0302, -0.0029,  0.0382,  0.0772,  0.0964,  0.0865,  0.0529,  0.0120, -0.0194;
//    expectedFilter2 = expectedFilter2 * 0.001;
//    ASSERT_EQ(filterSlice, expectedFilter2);

}

