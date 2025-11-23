// Copyright [2021] Optimus Ride Inc.

#pragma once

#include <cmath>
#include "altro/utils/assert.hpp"

/**
 * @brief Explicitly declare that a variable is unused
 * Supresses warnings for unused variables
 * 
 */
#define ALTRO_UNUSED(var) (void) (var)


inline double NormalizeAngle(double angle) {
    double a = std::fmod(angle + M_PI, 2.0 * M_PI);
    if (a < 0) a += 2.0 * M_PI;
    return a - M_PI;
}

inline double InterpolateAngle(double a0, double a1, double t) {
    double diff = NormalizeAngle(a1 - a0);
    return NormalizeAngle(a0 + t * diff);
}

template <typename T>
const T& safe_clamp(const T& value, const T& low, const T& high) {
    const T& real_low = (low < high) ? low : high;
    const T& real_high = (low < high) ? high : low;
    return (value < real_low) ? real_low : (value > real_high) ? real_high : value;
}