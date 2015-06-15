//
// Created by Adam Kosiorek on 09.05.15.
//

#ifndef NAME_UTILS_H
#define NAME_UTILS_H

#include <memory>
#include <cmath>
#include <glog/logging.h>

#include <boost/program_options.hpp>
#include <boost/timer/timer.hpp>


#define THROW_INVALID_ARG(msg) \
    LOG(ERROR) << msg; \
    throw std::invalid_argument(msg)

#define LOG_FUN         DLOG(INFO) << "Function:\t\t"            << __FUNCTION__
#define LOG_FUN_START   DLOG(INFO) << "Starting function:\t "    << __FUNCTION__
#define LOG_FUN_END     DLOG(INFO) << "Leaving function:\t"      << __FUNCTION__

template<class OutT = float, class InT>
OutT deg2rad(InT angle) {
    return static_cast<OutT>((M_PI * angle) / 180);
}

// add to std to be symmetric with make_shared
namespace std {
    template<class T, class... Ts>
    std::unique_ptr <T> make_unique(Ts... args) {
        return std::unique_ptr<T>(new T(std::forward<Ts>(args)...));
    }
} // namespace std

#endif //NAME_UTILS_H
