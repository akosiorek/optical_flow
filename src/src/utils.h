//
// Created by Adam Kosiorek on 09.05.15.
//

#ifndef NAME_UTILS_H
#define NAME_UTILS_H

#include <memory>
#include <glog/logging.h>

#define THROW_INVALID_ARG(msg) \
    LOG(ERROR) << msg; \
    throw std::invalid_argument(msg)



// add to std to be symmetric with make_shared
namespace std {
    template<class T, class... Ts>
    std::unique_ptr <T> make_unique(Ts... args) {
        return std::unique_ptr<T>(new T(args...));
    }
} // namespace std

#endif //NAME_UTILS_H
