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
template <typename float_t, typename Generator> float_t GenerateGaussianRandom(Generator& g, float_t mean, float_t sigma)
{
    std::normal_distribution<float_t> distribution(mean, sigma);
    return distribution(g);
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
    CrossEntropy_Sigmoid
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
    std::atomic<uint32_t> m_epochs_finished = 0;
    std::future<uint32_t> m_future;

    mutable std::atomic<bool> m_stop_at_next_epoch = false;
};
} // namespace macademy

#define ASSERTM(x, msg)                                                                                                                                                                                \
    if (!(x)) {                                                                                                                                                                                        \
        throw std::runtime_error("Assertion failed! " msg);                                                                                                                                            \
    }

#define ASSERT(x) ASSERTM(x, "")