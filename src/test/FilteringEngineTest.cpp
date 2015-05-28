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

struct FilterFactoryMock : public IFilterFactory {

    FilterFactoryMock(int filterSize) : filterSize_(filterSize) {}

    // create 1x1x1 filter with the only coefficient equal to angle
    virtual std::shared_ptr<Filter> createFilter(int angle) const override {

        auto filters = std::make_unique<std::vector<FilterT>>();

        filters->emplace_back(filterTransformer_(FilterT(filterSize_, filterSize_)));
        filters->at(0)(0, 0) = angle;
        return std::make_shared<Filter>(angle, std::move(filters));
    }

    virtual void setFilterTransformer(FilterTransformT transform) override {
        filterTransformer_ = transform;
    }

    int filterSize_;
    FilterTransformT filterTransformer_;
};

struct FourierTransformerMock : public IFourierTransformer {
    virtual void forward(const RealMatrix& src, ComplexMatrix& dst) const override {

    }

    virtual void backward(const ComplexMatrix& src, RealMatrix& dst) const override {

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

    void SetUp() {
        auto factory = std::make_unique<FilterFactoryMock>(filterSize_);
        auto padder = std::make_unique<FourierPadder>(dataSize_, filterSize_);
        auto transformer = std::make_unique<FourierTransformerMock>();
        engine = std::make_unique<EngineT>(std::move(factory), std::move(padder), std::move(transformer));
    }

    std::shared_ptr<EventQueueT> eventSliceQueue;
    std::shared_ptr<FlowQueueT> flowSliceQueue;
    std::unique_ptr<EngineT> engine;

    int dataSize_ = 2;

    //TODO change to 1 when fixed fourier padder arrives
    int filterSize_ = 2;
};


TEST_F(FilteringEngineTest, ConstructorPostconditionsTest) {

    ASSERT_FALSE(engine->isInitialized());
    ASSERT_FALSE(engine->hasInput());
    ASSERT_FALSE(engine->hasOutput());
    ASSERT_EQ(engine->timeSteps(), 0);
    ASSERT_EQ(engine->numFilters(), 0);
}

TEST_F(FilteringEngineTest, HasInputOutputTest) {

    engine->setInputBuffer(eventSliceQueue);
    engine->setOutputBuffer(flowSliceQueue);
    ASSERT_FALSE(engine->hasInput());
    ASSERT_FALSE(engine->hasOutput());

    eventSliceQueue->push(std::make_shared<EventSlice>());
    ASSERT_TRUE(engine->hasInput());
    ASSERT_FALSE(engine->hasOutput());

    flowSliceQueue->push(std::make_shared<FlowSlice>());
    ASSERT_TRUE(engine->hasInput());
    ASSERT_TRUE(engine->hasOutput());
}

TEST_F(FilteringEngineTest, AddFilterTest) {

    engine->addFilter(15);
    ASSERT_EQ(engine->numFilters(), 1);
    ASSERT_EQ(engine->timeSteps(), 1);

    //add the same filter
    engine->addFilter(15);
    ASSERT_EQ(engine->numFilters(), 1);

    //add another filter
    engine->addFilter(45);
    ASSERT_EQ(engine->numFilters(), 2);
}

TEST_F(FilteringEngineTest, IntializeAndProduceOutputTest) {
    engine->setOutputBuffer(flowSliceQueue);
    engine->addFilter(15);
    ASSERT_FALSE(engine->isInitialized());

    auto slice = std::make_shared<EventSlice>(dataSize_, dataSize_);
    engine->filter(slice);
    ASSERT_TRUE(engine->isInitialized());
    ASSERT_TRUE(engine->hasOutput());
}

TEST_F(FilteringEngineTest, ConsumeInputTest) {
    engine->setInputBuffer(eventSliceQueue);
    engine->setOutputBuffer(flowSliceQueue);
    engine->addFilter(15);
    ASSERT_FALSE(engine->isInitialized());

    auto slice = std::make_shared<EventSlice>(dataSize_, dataSize_);
    eventSliceQueue->push(slice);
    ASSERT_TRUE(engine->hasInput());
    engine->process();
    ASSERT_FALSE(engine->hasInput());
    ASSERT_TRUE(engine->isInitialized());
    ASSERT_TRUE(engine->hasOutput());
}


// TODO won't work after fourier transform get in place
TEST_F(FilteringEngineTest, FiltertTest) {
    int angle = 15;
    engine->setOutputBuffer(flowSliceQueue);
    engine->addFilter(angle);
    auto slice = std::make_shared<EventSlice>(dataSize_, dataSize_);
    slice->at(0, 0) = 1;
    slice->at(0, 1) = 2;
    slice->at(1, 0) = 3;
    slice->at(1, 1) = 4;
    engine->filter(slice);

    auto filtered = *flowSliceQueue->front();
    Eigen::MatrixXf expectedX(dataSize_, dataSize_);
    expectedX.setZero();
    expectedX(0, 0) = std::cos(deg2rad(angle)) * angle;
    Eigen::MatrixXf expectedY(dataSize_, dataSize_);
    expectedY.setZero();
    expectedY(0, 0) = -std::sin(deg2rad(angle)) * angle;

//    LOG(ERROR) << expectedX;
//    LOG(ERROR) << expectedY;
//    LOG(ERROR) << filtered.xv_;
//    LOG(ERROR) << filtered.yv_;
    ASSERT_TRUE(expectedX.isApprox(filtered.xv_));
    ASSERT_TRUE(expectedY.isApprox(filtered.yv_));
}




