//
// Created by dadrian on 5/14/2015.
//

uint32_t getNextPowerOfTwo(uint32_t n)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return ++v;
}