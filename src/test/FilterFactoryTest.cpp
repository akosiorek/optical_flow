/*
 * test.cpp
 *
 *  Created on: May 28, 2014
 *      Author: Adam Kosiorek
 */


#include "gtest/gtest.h"

#include "Filter.h"
#include "FilterFactory.h"
#include "../src/FilterFactory.h"

class FilterFactoryTest : public testing::Test {
public:
    int size(int radius) {
        return 2 * radius + 1;
    }

    float tolerance = 1e-6;
};

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
    FilterFactory factory(t0, tk, timeResolution, xRadius, yRadius);


    auto filter = factory.createFilter(angle);
    ASSERT_EQ(filter->angle(), angle);
    ASSERT_EQ(filter->numSlices(), (tk - t0) / timeResolution);
    ASSERT_EQ(filter->xSize(), this->size(xRadius));
    ASSERT_EQ(filter->ySize(), this->size(yRadius));
    auto filterSlice = filter->at(0);

    // prepare values we expect the filter to have
    decltype(filterSlice) expectedFilter(this->size(xRadius), this->size(yRadius));
    expectedFilter << 0.043689713870756, 0.041171961916116, 0.020679854734440, -0.013160985223100, -0.047270790019091,
            -0.067110772224775, -0.065325844480389, -0.045639528855475, -0.019404156399499, 0.041171961916116,
            0.022028244995357, -0.014933211980382, -0.057133404440230, -0.086401621554250, -0.089587435336400,
            -0.066670795661262, -0.030194070610011, 0.003217216792724, 0.020679854734440, -0.014933211980382,
            -0.060858678486375, -0.098036257031402, -0.108279027572214, -0.085835174659464, -0.041407950709754,
            0.004699750605773, 0.034948928833061, -0.013160985223100, -0.057133404440230, -0.098036257031402,
            -0.115339153869756, -0.097393533754073, -0.050047337774259, 0.006050683962916, 0.047928732139823,
            0.063197302795920, -0.047270790019091, -0.086401621554250, -0.108279027572214, -0.097393533754073,
            -0.053310578435580, 0.006865454577513, 0.057928620117118, 0.081363233627144, 0.074659312950490,
            -0.067110772224775, -0.089587435336400, -0.085835174659464, -0.050047337774259, 0.006865454577513,
            0.061705744676132, 0.092319411850024, 0.090236290112929, 0.063583617471466, -0.065325844480389,
            -0.066670795661262, -0.041407950709754, 0.006050683962916, 0.057928620117118, 0.092319411850024,
            0.096119974323096, 0.072145635154603, 0.035564784161392, -0.045639528855475, -0.030194070610011,
            0.004699750605773, 0.047928732139823, 0.081363233627144, 0.090236290112929, 0.072145635154603,
            0.037883717694081, 0.003838842099420, -0.019404156399499, 0.003217216792724, 0.034948928833061,
            0.063197302795920, 0.074659312950490, 0.063583617471466, 0.035564784161392, 0.003838842099420,
            -0.018952364291003;

    expectedFilter = expectedFilter * 0.001;

    for (int x = 0; x < this->size(xRadius); ++x) {
        for (int y = 0; y < this->size(yRadius); ++y) {
            ASSERT_NEAR(filterSlice(x, y), expectedFilter(x, y), tolerance) << "x = " << x << ", y = " << y;
        }
    }
}

TEST_F(FilterFactoryTest, TransformedFilterTest) {

    int angle = 45;
    float t0 = 0;
    float tk = 0.01;
    float timeResolution = 0.01;
    int xRadius = 1;
    int yRadius = 1;
    FilterFactory factory(t0, tk, timeResolution, xRadius, yRadius);

    auto transformFun = [](const Eigen::MatrixXf& filter) -> Eigen::MatrixXf {
        return filter / filter.maxCoeff();
    };
    factory.setFilterTransformer(transformFun);

    auto filterbank = factory.createFilter(angle);
    auto filter = filterbank->at(0);
    ASSERT_NEAR(filter.maxCoeff(), 1, this->tolerance);
}

