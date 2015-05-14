/*
 * test.cpp
 *
 *  Created on: May 28, 2014
 *      Author: Adam Kosiorek
 */


#include "gtest/gtest.h"

#include "Filter.h"
#include "FilterFactory.h"

class FactoryFilterTest : public testing::Test {};


// similar for another angle, at least 3 time slices, odd and even ranges
TEST_F(FactoryFilterTest, DISABLED_SmallFilterTest) {

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

