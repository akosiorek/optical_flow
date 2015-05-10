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

#include "../src/BlockingQueue.h"


class BlockingQueueTest : public testing::Test {

    void SetUp() {
        bqueue = make_unique<BlockingQueue<int> >();
    }

public:
    std::unique_ptr<BlockingQueue<int> > bqueue;
};

TEST_F(BlockingQueueTest, EmptyTest)
{
    ASSERT_EQ(bqueue->size(), 0);
}

TEST_F(BlockingQueueTest, AddElementTest)
{
    bqueue->push(4);
    ASSERT_EQ(bqueue->size(), 1);
    bqueue->push(3);
    bqueue->push(4);
    ASSERT_EQ(bqueue->size(), 3);
}

TEST_F(BlockingQueueTest, PopElementTest)
{
    bqueue->push(4);
    ASSERT_EQ(bqueue->size(), 1);
    bqueue->push(3);
    bqueue->push(4);
    ASSERT_EQ(bqueue->size(), 3);
    bqueue->pop();
    ASSERT_EQ(bqueue->size(), 2);
    bqueue->pop();
    ASSERT_EQ(bqueue->size(), 1);      
    bqueue->pop();
    ASSERT_EQ(bqueue->size(), 0);    
    bqueue->pop();
}

TEST_F(BlockingQueueTest, FrontWriteTest)
{
    bqueue->push(4);
    bqueue->push(3);
    bqueue->push(2);
    int& fe = bqueue->front();
    ASSERT_EQ(bqueue->front(), 4);
    fe = 5;
    ASSERT_EQ(bqueue->front(), 5);    
    bqueue->pop();
    ASSERT_EQ(bqueue->front(), 3);
}

TEST_F(BlockingQueueTest, BackWriteTest)
{
    bqueue->push(4);
    bqueue->push(3);
    bqueue->push(2);
    int& be = bqueue->back();
    ASSERT_EQ(bqueue->back(), 2);
    be = 7;
    ASSERT_EQ(bqueue->back(), 7);    
    bqueue->pop();
    ASSERT_EQ(bqueue->back(), 7);
}