/*
 * test.cpp
 *
 *  Created on: May 28, 2014
 *      Author: Adam Kosiorek
 */

#include <memory>
#include "gtest/gtest.h"

#include "utils.h"

#include "Quantizer.h"



class QuantizerTest : public testing::Test {

    void SetUp() {
//        quantizer = std::unique_ptr<Quantizer>(new Quantizer());
    }

public:
    std::unique_ptr<Quantizer> quantizer;


};

TEST_F(QuantizerTest, ConstructorTest) {

    quantizer = make_unique<Quantizer>(2);
    ASSERT_EQ(quantizer->getTimeResolution(), 2);
    ASSERT_EQ(quantizer->currentTimeStep_, 0);
}



TEST_F(QuantizerTest, QuantizeTest) {

    quantizer = make_unique<Quantizer>(2);
    std::vector<Event> events = {{1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}};
    quantizer->quantize(events);


    auto event = quantizer->getEventSlice();
    ASSERT_EQ(event(1, 1), 1);
    ASSERT_EQ(event(2, 2), 2);
    ASSERT_EQ(event(127, 127), 0);

    event = quantizer->getEventSlice();
    ASSERT_EQ(event(1, 1), 0);
    ASSERT_EQ(event(3, 3), 3);

    // throw if empty
    ASSERT_THROW(quantizer->getEventSlice(), std::runtime_error);

//    GTEST_ASSERT_EQ(quantizer->getEventSlices().size(), 2);


}

