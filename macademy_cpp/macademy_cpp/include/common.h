#pragma once

#include <string>
#include <vector>
#include <memory>
#include <span>
#include <cstdint>
#include <stdint.h>
#include <random>
#include <stdexcept>
#include <future>
#include <atomic>

namespace macademy {
template <typename float_t> float_t GenerateGaussianRandom(float_t mean, float_t sigma)
{
    thread_local std::default_random_engine generator;

    std::normal_distribution<float_t> distribution(mean, sigma);
    return distribution(generator);
}

enum class ActivationFunction
{
    Passtrough,
    Sigmoid,
    ReLU
};

enum class CostFunction
{
    MeanSquared,
    CrossEntropy
};

enum class Regularization
{
    None,
    L1,
    L2
};

struct TrainingResultTracker
{
    std::atomic<float> m_epoch_progress = 0;
    std::atomic<uint64_t> m_epochs_finished = 0;
    std::atomic<bool> m_stop_at_next_epoch = false;
    std::future<int64_t> m_future;
};
} // namespace macademy

#define ASSERTM(x, msg)                                                                                                                                                                                \
    if (!(x)) {                                                                                                                                                                                        \
        throw std::runtime_error("Assertion failed! " msg);                                                                                                                                            \
    }

#define ASSERT(x) ASSERTM(x, "")