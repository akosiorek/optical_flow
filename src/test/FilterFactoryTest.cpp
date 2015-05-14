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
    expectedFilter<< 0.043689713870756, 0.041171961916116, 0.020679854734440, -0.013160985223100, -0.047270790019091, -0.067110772224775, -0.065325844480389, -0.045639528855475, -0.019404156399499, 0.041171961916116, 0.022028244995357, -0.014933211980382, -0.057133404440230, -0.086401621554250, -0.089587435336400, -0.066670795661262, -0.030194070610011, 0.003217216792724, 0.020679854734440, -0.014933211980382, -0.060858678486375, -0.098036257031402, -0.108279027572214, -0.085835174659464, -0.041407950709754, 0.004699750605773, 0.034948928833061, -0.013160985223100, -0.057133404440230, -0.098036257031402, -0.115339153869756, -0.097393533754073, -0.050047337774259, 0.006050683962916, 0.047928732139823, 0.063197302795920, -0.047270790019091, -0.086401621554250, -0.108279027572214, -0.097393533754073, -0.053310578435580, 0.006865454577513, 0.057928620117118, 0.081363233627144, 0.074659312950490, -0.067110772224775, -0.089587435336400, -0.085835174659464, -0.050047337774259, 0.006865454577513, 0.061705744676132, 0.092319411850024, 0.090236290112929, 0.063583617471466, -0.065325844480389, -0.066670795661262, -0.041407950709754, 0.006050683962916, 0.057928620117118, 0.092319411850024, 0.096119974323096, 0.072145635154603, 0.035564784161392, -0.045639528855475, -0.030194070610011, 0.004699750605773, 0.047928732139823, 0.081363233627144, 0.090236290112929, 0.072145635154603, 0.037883717694081, 0.003838842099420, -0.019404156399499, 0.003217216792724, 0.034948928833061, 0.063197302795920, 0.074659312950490, 0.063583617471466, 0.035564784161392, 0.003838842099420, -0.018952364291003;
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

