//
// Created by Adam Kosiorek on 09.05.15.
//

#ifndef NAME_UTILS_H
#define NAME_UTILS_H

#include <memory>

template<class T, class... Ts>
std::unique_ptr<T> make_unique(Ts... args) {
    return std::unique_ptr<T>(new T(args...));
}

#endif //NAME_UTILS_H
