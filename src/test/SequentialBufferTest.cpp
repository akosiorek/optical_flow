/*
 * test.cpp
 *
 *  Created on: May 28, 2014
 *      Author: Adam Kosiorek
 */


#include "gtest/gtest.h"
#include <queue>

class Event{
public:
//    Event(){
//        x=0;
//        y=0;
//        time=0;
//        edge=0;
//
//    };

    Event(unsigned int x, unsigned int y, unsigned time, int edge):x_(x),y_(y),time_(time),edge_(edge){
    };
    Event():Event(0,0,0,0){
    };

    unsigned int x_;
    unsigned int y_;
    unsigned int time_;
    int edge_;
};

class SequentialBuffer{
public:
    SequentialBuffer(){
    };
    void push(Event event);
    Event pop();

    std::queue<Event> eventQueue_;

//    void init();

};

void SequentialBuffer::push(Event event) {
    eventQueue_.push(event);

}

Event SequentialBuffer::pop(){
    Event res =eventQueue_.front();
    eventQueue_.pop();
    return res;
}

//void SequentialBuffer::init(){
//
//}


class EventTest : public testing::Test {


};


TEST_F(EventTest, DefaultConstructorTest) {
    Event e;
    GTEST_ASSERT_EQ(e.x_, 0);
    GTEST_ASSERT_EQ(e.y_, 0);
    GTEST_ASSERT_EQ(e.time_, 0);
    GTEST_ASSERT_EQ(e.edge_, 0);
    Event e2 = Event(1,1,1,1);
    GTEST_ASSERT_EQ(e2.x_, 1);
    GTEST_ASSERT_EQ(e2.y_, 1);
    GTEST_ASSERT_EQ(e2.time_, 1);
    GTEST_ASSERT_EQ(e2.edge_, 1);


}

class SequentialBufferTest : public testing::Test {


};


TEST_F(SequentialBufferTest, SomeTest) {
    Event e;
    Event e2 = Event(1,1,1,1);

    SequentialBuffer b;
    b.push(e);
    b.push(e2);
    GTEST_ASSERT_EQ(b.pop().x_, e.x_);


}

