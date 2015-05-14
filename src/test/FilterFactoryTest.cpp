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
TEST_F(FilterFactoryTest, DISABLED_SmallFilterTest) {

    int angle = 45;
    float t0 = 0;
    float tk = 0.01;
    float timeResolution = 0.01;
    int xRange = 1;
    int yRange = 1;
    FilterFactory factory(t0, tk, timeResolution, xRange, yRange);


    auto filter = factory.createFilter(angle);
    ASSERT_EQ(filter->angle(), angle);
    ASSERT_EQ(filter->numSlices(), (tk - t0) /timeResolution);
    ASSERT_EQ(filter->xSize(), 2 * xRange + 1);
    ASSERT_EQ(filter->ySize(), 2 * yRange + 1);
    auto filterSlice = filter->at(0);

    // prepare values we expect the filter to have
    decltype(filterSlice) expectedFilter = {};

    ASSERT_EQ(filterSlice, expectedFilter);
}

