//
// Created by Adam Kosiorek on 09.05.15.
//

#ifndef OPTICAL_FLOW_EVENT_H
#define OPTICAL_FLOW_EVENT_H

class Event {
public:

    Event(unsigned int x, unsigned int y, unsigned time, int edge)
            : x_(x),y_(y),time_(time),edge_(edge) {};

    Event()
            :Event(0,0,0,0) {};

    unsigned int x_;
    unsigned int y_;
    unsigned int time_;
    int edge_;
};

#endif //OPTICAL_FLOW_EVENT_H
