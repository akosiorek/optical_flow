/*
 * test.cpp
 *
 *  Created on: May 28, 2014
 *      Author: Adam Kosiorek
 */


#include "gtest/gtest.h"
#include "common.h"
#include "BlockingQueue.h"
#include "Quantizer.h"
#include "EventSlice.h"

class QuantizerTest : public testing::Test {

    void SetUp() {
        quantizer = std::make_unique<Quantizer<BlockingQueue>>(2);
    }

public:
    std::unique_ptr<Quantizer<BlockingQueue>> quantizer;
};

TEST_F(QuantizerTest, ConstructorTest) {

    ASSERT_EQ(quantizer->getTimeResolution(), 2);
    ASSERT_EQ(quantizer->getCurrentTimeStep(), 0);
}

TEST_F(QuantizerTest, NoEventTest) {

    ASSERT_TRUE(quantizer->isEmpty());
    quantizer->quantize(std::vector<Event>());
    ASSERT_TRUE(quantizer->isEmpty());
    ASSERT_TRUE(quantizer->getEventSlice().isZero());
}

TEST_F(QuantizerTest, SingleEventTest) {

    // if time resolution > 1 there's not event slice with a single event.
    // Another event, with a time + timeResolution, would have to be passed
    // to generate an event slice.
    std::vector<Event> events = {{1, 1, 1, 1, 0}};

    quantizer->quantize(events);
    ASSERT_TRUE(quantizer->isEmpty());

    auto eventSlice = quantizer->getEventSlice();
    ASSERT_TRUE(eventSlice.isZero());

    events = {{4, 1, 2, 0, 0}};
    quantizer->quantize(events);
    ASSERT_FALSE(quantizer->isEmpty());
    eventSlice = quantizer->getEventSlice();
    ASSERT_EQ(eventSlice(1, 1), 1);
}

TEST_F(QuantizerTest, QuantizeTest) {
    
    std::vector<Event> events = {{1, 1, 1, 1, 0}, 
                                 {2, 2, 2, 0, 0}, 
                                 {3, 3, 3, 0, 0}, 
                                 {5, 5, 5, 1, 0}};
               
    quantizer->quantize(events);
    ASSERT_FALSE(quantizer->isEmpty());

    auto event = quantizer->getEventSlice();
    ASSERT_FALSE(quantizer->isEmpty());
    ASSERT_EQ(event(1, 1), 1);
    ASSERT_EQ(event(2, 2), -1);
    ASSERT_EQ(event(127, 127), 0);

    event = quantizer->getEventSlice();
    ASSERT_TRUE(quantizer->isEmpty());
    ASSERT_EQ(event(1, 1), 0);
    ASSERT_EQ(event(3, 3), -1);

    ASSERT_TRUE(quantizer->getEventSlice().isZero());
}

TEST_F(QuantizerTest, QuantizeGetMultipleEventsTest) {

    quantizer = std::make_unique<Quantizer<BlockingQueue>>(3);
    std::vector<Event> events = {
            // time, x, y, parity, id
            {1, 1, 1, 1, 0},
            {2, 2, 2, 0, 0},
            {3, 3, 3, 1, 0},
            {4, 4, 4, 0, 0},
            {5, 5, 5, 1, 0},
            {9, 6, 6, 0, 0},
            {10, 7, 7, 1, 0},
            {11, 7, 7, 1, 0},
            {12, 9, 9, 1, 0},
            {15, 10, 10, 0, 0}
    };

    std::vector<std::vector<int>> expectedEntries = {
            // event, x, y, parity
            {0, 1, 1, 1},
            {0, 2, 2, -1},
            {0, 3, 3, 1},
            {1, 4, 4, -1},
            {1, 5, 5, 1},
            {2, 6, 6, -1},
            {3, 7, 7, 2},
            {3, 9, 9, 1},
            {4, 10, 10, -1}
    };

    quantizer->quantize(events);

    auto eventSlices = quantizer->getEventSlices();
    ASSERT_TRUE(quantizer->isEmpty());
    ASSERT_EQ(eventSlices->size(), 5);

    auto last_slice = expectedEntries[0][0];
    for(const auto& expected : expectedEntries) {
        auto slice = expected[0];
        auto x = expected[1];
        auto y = expected[2];
        auto parity = expected[3];
        if(last_slice!=slice)
        {
            eventSlices->pop();
            last_slice=slice;
        }
        ASSERT_EQ((*eventSlices->front())(x, y), parity) << "slice = " << slice << ", x = " << x << ", y = " << y;
    }
}

TEST_F(QuantizerTest, LongPauseBetweenEventsTest) {

    std::vector<Event> events = {
            {1, 1, 1, 1, 0},
            {100, 1, 1, 1, 0},
            {200, 1, 1, 1, 0}
    };

    quantizer->quantize(events);
    ASSERT_FALSE(quantizer->isEmpty());
    auto slices = quantizer->getEventSlices();
    ASSERT_TRUE(quantizer->isEmpty());
    ASSERT_EQ(slices->size(), 100);
    std::vector<std::shared_ptr<EventSlice>> nonZero;

    while(!slices->empty())
    {
        if(!(*slices->front()).isZero()) nonZero.push_back(slices->front());
        slices->pop();
    }

    //  = {slices->at(0), slices->at(49), slices->at(99)};
    // slices->erase(slices->begin()+99);
    // slices->erase(slices->begin()+49);
    // slices->erase(slices->begin());

    for(auto& e : nonZero) {
        ASSERT_EQ((*e)(1, 1), 1);
        (*e)(1, 1) = 0;
        ASSERT_TRUE(e->isZero());
    }
}
