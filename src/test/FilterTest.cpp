/*
 * test.cpp
 *
 *  Created on: May 28, 2014
 *      Author: Adam Kosiorek
 */

#include <memory>
#include <glog/logging.h>
#include "gtest/gtest.h"
#include "utils.h"

#include "Filter.h"

class FilterTest : public testing::Test {

    void SetUp(){
//        filterDefault=make_unique<Filter>();
        filterDefault=Filter();
        filterSlices=std::make_unique<std::vector<Eigen::MatrixXf>>(3);
        filterSlices->at(0)=Eigen::MatrixXf::Random(3,3);
        filterSlices->at(1)=Eigen::MatrixXf::Random(3,3);
        filterSlices->at(2)=Eigen::MatrixXf::Random(3,3);


    }

public:
    Filter filterDefault;
    Filter filter;
    std::unique_ptr<std::vector<Eigen::MatrixXf>> filterSlices;

};


TEST_F(FilterTest, DefaultConstructorTest) {

    ASSERT_EQ(filterDefault.xSize(),0);
    ASSERT_EQ(filterDefault.ySize(),0);
    ASSERT_EQ(filterDefault.angle(),0);
//    ASSERT_EQ(filter->filters_,nullptr);


}

TEST_F(FilterTest, ConstructorTest) {
    int angle =45;
    ASSERT_NE(filterSlices,nullptr);
    
    filter= Filter(angle,std::move(filterSlices));

    ASSERT_EQ(filter.xSize(),3);
    ASSERT_EQ(filter.ySize(),3);
    ASSERT_EQ(filter.angle(),45);
    ASSERT_EQ(filterSlices,nullptr);
}
