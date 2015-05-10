//
// Created by Adam Kosiorek on 09.05.15.
//

#ifndef OPTICAL_FLOW_EVENT_H
#define OPTICAL_FLOW_EVENT_H

#include <cstdint>

class Event {
public:
    typedef uint64_t TimeT;

    Event(int x, int y, TimeT time, int edge)
            : x_(x),y_(y),time_(time),edge_(edge) {};

    Event()
            :Event(0,0,0,0) {};

    int x_;
    int y_;
    TimeT time_;
    int edge_;
};

#endif //OPTICAL_FLOW_EVENT_H
