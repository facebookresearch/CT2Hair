// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
// 
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <iostream>
#include <ctime>
#include <ratio>
#include <chrono>
#include <iomanip>

class Clock {
public:
    Clock() {}
    ~Clock() {}

    void tick() {
        clock_begin = std::chrono::steady_clock::now();
    }

    double tock() {
        std::chrono::steady_clock::time_point clock_end = std::chrono::steady_clock::now();
        double sec = (double)(std::chrono::duration_cast<std::chrono::milliseconds>(clock_end - clock_begin).count()) / 1000.f;
        return sec;
    }

    void print() {
        std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
        std::time_t now_c = std::chrono::system_clock::to_time_t(now);
        std::cout << "Current time: " << std::put_time(std::localtime(&now_c), "%F %T") << std::endl;
    };

private:
    std::chrono::steady_clock::time_point clock_begin;

};