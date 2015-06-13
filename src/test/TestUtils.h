//
// Created by Adam Kosiorek on 6/13/15.
//

#ifndef OPTICAL_FLOW_TESTUTILS_H
#define OPTICAL_FLOW_TESTUTILS_H

#define ASSERT_NEAR_VEC(x, y, N) for(std::size_t i = 0; i < N; ++i) ASSERT_NEAR(x[i], y[i], this->tolerance)
#define ASSERT_NEAR_VEC_COMPLEX(x, y, N) \
    for(std::size_t i = 0; i < N; ++i) ASSERT_NEAR(x[i].real(), y[i].real(), this->tolerance); \
    for(std::size_t i = 0; i < N; ++i) ASSERT_NEAR(x[i].imag(), y[i].imag(), this->tolerance)

#endif //OPTICAL_FLOW_TESTUTILS_H
