/*
 * test.cpp
 *
 *  Created on: May 28, 2014
 *      Author: Adam Kosiorek
 */

#include "gtest/gtest.h"

#include "common.h"

struct Filter {};

template<template <class> class InputBufferType, template <class> class OutputBufferType = InputBufferType>
class FilteringEngine {
public:

    void loadFilters(const std::string& filterURI);
    std::vector<std::shared_ptr<Filter>> getFilters();
    std::shared_ptr<FilterSlice> filter(std::shared_ptr<EventSlice> eventSlice);
    void process();


    void setInputBuffer(std::shared_ptr<InputBufferType<EventSlice>> buffer);
    void setOutputBuffer(std::shared_ptr<OutputBufferType<FlowSlice>> buffer);

private:
};

class FilteringEngineTest : public testing::Test {
public:

    void SetUp() {
        engine.setInputBuffer(eventSliceQueue);
        engine.setOutputBuffer(flowSliceQueue);
    }

    BlockingQueue<EventSlice> eventSliceQueue;
    BlockingQueue<FlowSlice> flowSliceQueue;
    FilteringEngine<BlockingQueue> engine;
};


TEST_F(FilteringEngineTest, LoadFilterTest) {

    std::string smallFilterPath = "../data/loadFilterTest.mat";
    std::vector<float> expectedFilter = {0, 0, 0};


    engine.loadFilters(smallFilterPath);
    auto filter = engine.getFilters()[0];

    ASSERT_EQ(filter->size(), expectedFilter.size());
    for(int i = 0; i < expectedFilter.size(); ++i) {
        ASSERT_EQ(filter->at(i), expectedFilter[i]) << "Failed at i = " << i;
    };
}

TEST_F(FilteringEngineTest, DirectFilterTest) {

    std::vector<float> expectedResult = {1, 2, 3};
    std::string filterURI = "../data/directFilterTest.mat";

    auto slice = std::make_shared<EventSlice>();
    std::vector<Eigen::Triplet<int>> triplets = {{1, 1, 1}, {2, 3, 4}};
    slice->setFromTriplets(tripletrs.begin(), triplets.end());

    engine.loadFilters(filterURI);
    auto flow = engine.filter(slice);

    ASSERT_EQ(flow->size(), expectedResult.size());
    for(int i = 0; i < expectedResult.size(); ++i) {
        ASSERT_EQ(flow->at(i), expectedResult[i]) << "Failed at i = " << i;
    };
}

TEST_F(FilteringEngineTest, SingleEventSliceInInputBufferTest) {

    std::vector<float> expectedResult = {1, 2, 3};
    std::string filterURI = "../data/directFilterTest.mat";

    EventSlice slice;
    std::vector<Eigen::Triplet<int>> triplets = {{1, 1, 1}, {2, 3, 4}};
    slice.setFromTriplets(tripletrs.begin(), triplets.end());

    engine.loadFilter(filterURI);

    ASSET_TRUE(eventSliceQueue.empty());
    ASSERT_TRUE(flowSliceQueue.empty());

    eventSliceQueue.push(slice);
    engine.process();

    ASSET_TRUE(eventSliceQueue.empty());
    ASSERT_EQ(flowSliceQueue.size(), 1);

    auto flow = flowSliceQueue.front();

    ASSERT_EQ(flow.size(), expectedResult.size());
    for(int i = 0; i < expectedResult.size(); ++i) {
        ASSERT_EQ(flow[i], expectedResult[i]) << "Failed at i = " << i;
    };
}

TEST_F(FilteringEngineTest, ManyEventSlicesInInputBufferTest) {

    std::vector<float> expectedResult = {1, 2, 3};
    std::string filterURI = "../data/directFilterTest.mat";

    EventSlice slice;
    std::vector<Eigen::Triplet<int>> triplets = {{1, 1, 1}, {2, 3, 4}};
    slice.setFromTriplets(tripletrs.begin(), triplets.end());

    engine.loadFilter(filterURI);

    ASSET_TRUE(eventSliceQueue.empty());
    ASSERT_TRUE(flowSliceQueue.empty());

    eventSliceQueue.push(slice);
    eventSliceQueue.push(slice);
    eventSliceQueue.push(slice);

    engine.process();

    ASSET_TRUE(eventSliceQueue.empty());
    ASSERT_EQ(flowSliceQueue.size(), 3);

    while(!flowSliceQueue.empty()) {
        auto flow = flowSliceQueue.front();
        flowSliceQueue.pop();

        ASSERT_EQ(flow.size(), expectedResult.size());
        for (int i = 0; i < expectedResult.size(); ++i) {
            ASSERT_EQ(flow[i], expectedResult[i]) << "Failed at i = " << i;
        };
    }
}




