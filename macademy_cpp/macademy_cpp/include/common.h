#pragma once

#include <string>
#include <vector>
#include <memory>
#include <span>
#include <cstdint>
#include <stdint.h>
#include <random>

namespace macademy
{
    template <typename float_t>
    float_t GenerateGaussianRandom(float_t mean, float_t sigma)
    {
        thread_local std::default_random_engine generator;

        std::normal_distribution<float_t> distribution(mean, sigma);
        return distribution(generator);
    }

}

#define ASSERTM(x, msg)                                                                                                                                                                                \
    if (!(x)) {                                                                                                                                                                                        \
        throw std::runtime_error("Assertion failed! " msg);                                                                                                                                            \
    }