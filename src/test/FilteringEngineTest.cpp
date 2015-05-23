/*
 * test.cpp
 *
 *  Created on: May 28, 2014
 *      Author: Adam Kosiorek
 */

#include "gtest/gtest.h"

#include "utils.h"
#include "FilterFactory.h"
#include "BlockingQueue.h"
#include "IFilterFactory.h"
#include "Filter.h"
#include "EventSlice.h"
#include "FlowSlice.h"
#include "FilteringEngine.h"
#include "../src/IFilterFactory.h"

struct FilterFactoryMock : IFilterFactory {
    // create 1x1x1 filter with the only coefficient equal to angle
    virtual std::shared_ptr<Filter> createFilter(int angle) const override {

        auto filters = std::make_unique<std::vector<Eigen::MatrixXf>>();
        filters->emplace_back(1, 1);
        filters->at(0)(0, 0) = angle;
        return std::make_shared<Filter>(angle, std::move(filters));
    }
};

class FilteringEngineTest : public testing::Test {
public:
    typedef FilteringEngine<BlockingQueue> EngineT;
    typedef typename EngineT::EventQueueT EventQueueT;
    typedef typename EngineT::FlowQueueT FlowQueueT;

    FilteringEngineTest() :
            eventSliceQueue(new EventQueueT()),
            flowSliceQueue(new FlowQueueT()) {}

    std::shared_ptr<EventQueueT> eventSliceQueue;
    std::shared_ptr<FlowQueueT> flowSliceQueue;
    std::unique_ptr<EngineT> engine;
};


TEST_F(FilteringEngineTest, InitializeTest) {

    auto factory = std::make_unique<FilterFactoryMock>();
    engine = std::make_unique<EngineT>(std::move(factory));
    engine->setInputBuffer(eventSliceQueue);
    engine->setOutputBuffer(flowSliceQueue);
    ASSERT_EQ(engine->timeSteps(), 0);

    engine->addFilter(15);
    ASSERT_EQ(engine->timeSteps(), 1);
    ASSERT_FALSE(engine->isInitialized());
    ASSERT_FALSE(engine->hasOutput());

    auto slice = std::make_shared<EventSlice>();
    engine->filter(slice);
    ASSERT_TRUE(engine->isInitialized());
    ASSERT_FALSE(engine->hasOutput());

    auto slice2 = std::make_shared<EventSlice>();
    engine->filter(slice2);
    ASSERT_TRUE(engine->hasOutput());
}

//TEST_F(FilteringEngineTest, DirectFilterTest) {
//
//    std::vector<float> expectedResult = {1, 2, 3};
//    std::string filterURI = "../data/directFilterTest.mat";
//
//    auto slice = std::make_shared<EventSlice>();
//    std::vector<Eigen::Triplet<int>> triplets = {{1, 1, 1}, {2, 3, 4}};
//    slice->setFromTriplets(tripletrs.begin(), triplets.end());
//
//    engine.loadFilters(filterURI);
//    auto flow = engine.filter(slice);
//
//    ASSERT_EQ(flow->size(), expectedResult.size());
//    for(int i = 0; i < expectedResult.size(); ++i) {
//        ASSERT_EQ(flow->at(i), expectedResult[i]) << "Failed at i = " << i;
//    };
//}
//
//TEST_F(FilteringEngineTest, SingleEventSliceInInputBufferTest) {
//
//    std::vector<float> expectedResult = {1, 2, 3};
//    std::string filterURI = "../data/directFilterTest.mat";
//
//    EventSlice slice;
//    std::vector<Eigen::Triplet<int>> triplets = {{1, 1, 1}, {2, 3, 4}};
//    slice.setFromTriplets(tripletrs.begin(), triplets.end());
//
//    engine.loadFilter(filterURI);
//
//    ASSET_TRUE(eventSliceQueue.empty());
//    ASSERT_TRUE(flowSliceQueue.empty());
//
//    eventSliceQueue.push(slice);
//    engine.process();
//
//    ASSET_TRUE(eventSliceQueue.empty());
//    ASSERT_EQ(flowSliceQueue.size(), 1);
//
//    auto flow = flowSliceQueue.front();
//
//    ASSERT_EQ(flow.size(), expectedResult.size());
//    for(int i = 0; i < expectedResult.size(); ++i) {
//        ASSERT_EQ(flow[i], expectedResult[i]) << "Failed at i = " << i;
//    };
//}
//
//TEST_F(FilteringEngineTest, ManyEventSlicesInInputBufferTest) {
//
//    std::vector<float> expectedResult = {1, 2, 3};
//    std::string filterURI = "../data/directFilterTest.mat";
//
//    EventSlice slice;
//    std::vector<Eigen::Triplet<int>> triplets = {{1, 1, 1}, {2, 3, 4}};
//    slice.setFromTriplets(tripletrs.begin(), triplets.end());
//
//    engine.loadFilter(filterURI);
//
//    ASSET_TRUE(eventSliceQueue.empty());
//    ASSERT_TRUE(flowSliceQueue.empty());
//
//    eventSliceQueue.push(slice);
//    eventSliceQueue.push(slice);
//    eventSliceQueue.push(slice);
//
//    engine.process();
//
//    ASSET_TRUE(eventSliceQueue.empty());
//    ASSERT_EQ(flowSliceQueue.size(), 3);
//
//    while(!flowSliceQueue.empty()) {
//        auto flow = flowSliceQueue.front();
//        flowSliceQueue.pop();
//
//        ASSERT_EQ(flow.size(), expectedResult.size());
//        for (int i = 0; i < expectedResult.size(); ++i) {
//            ASSERT_EQ(flow[i], expectedResult[i]) << "Failed at i = " << i;
//        };
//    }
//}




