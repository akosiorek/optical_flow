/*
 * test.cpp
 *
 *  Created on: May 28, 2014
 *      Author: Adam Kosiorek
 */

#include <memory>

#include "gtest/gtest.h"

class Quantizer {
public:

 int method() {return 0;};

};

class QuantizerTest : public testing::Test {

    void SetUp() {

        quantizer = std::unique_ptr<Quantizer>(new Quantizer());
    }

public:
    std::unique_ptr<Quantizer> quantizer;
};


TEST_F(QuantizerTest, SomeTest) {


    GTEST_ASSERT_EQ(this->quantizer->method(), 1);
}

