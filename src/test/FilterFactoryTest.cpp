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
    expectedFilter<< 0.057811657300327, 0.057971259127962, 0.034529629086888, -0.008103576254342, -0.053837452436498,
            -0.083319201360517, -0.085377654864692, -0.063059441967883, -0.030152450743285, 0.057971259127962, 0.036781067318526,
            -0.009194784429426, -0.065070140415335, -0.107269129314178, -0.117086356788855, -0.092118023024893, -0.046919083111035,
            -0.002861314830458, 0.034529629086888, -0.009194784429426, -0.069312914106895, -0.121713733420808, -0.141515345399328,
            -0.118597153629298, -0.064344523330500, -0.004179844559485, 0.038174445720664, -0.008103576254342, -0.065070140415335,
            -0.121713733420808, -0.150742582048578, -0.134567162366210, -0.077769414758471, -0.005381332024829, 0.052352184877298,
            0.077237994233577, -0.053837452436498, -0.107269129314178, -0.141515345399328, -0.134567162366210, -0.082840220274475,
            -0.006105969310151, 0.063274985476578, 0.099439892079132, 0.096396946991059, -0.083319201360517, -0.117086356788855,
            -0.118597153629298, -0.077769414758471, -0.006105969310151, 0.067400709533730, 0.112830229846134, 0.116509281038445,
            0.086456167519524, -0.085377654864692, -0.092118023024893, -0.064344523330500, -0.005381332024829, 0.063274985476578,
            0.112830229846134, 0.124106045226400, 0.098098148025754, 0.052869877955025, -0.063059441967883, -0.046919083111035,
            -0.004179844559485, 0.052352184877298, 0.099439892079132, 0.116509281038445, 0.098098148025754, 0.056317156934781,
            0.011971799619448, -0.030152450743285, -0.002861314830458, 0.038174445720664, 0.077237994233577, 0.096396946991059,
            0.086456167519524, 0.052869877955025, 0.011971799619448, -0.019371866393356;
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

