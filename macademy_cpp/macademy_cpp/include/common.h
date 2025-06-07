#pragma once

#include <string>
#include <cstring>
#include <vector>
#include <memory>
#include <span>
#include <cstdint>
#include <stdint.h>
#include <random>
#include <stdexcept>
#include <future>
#include <atomic>
#include <nlohmann/json.hpp>

namespace macademy {
template <typename float_t, typename Generator> float_t GenerateGaussianRandom(Generator& g, float_t mean, float_t sigma)
{
    std::normal_distribution<float_t> distribution(mean, sigma);
    return distribution(g);
}

enum class ActivationFunction
{
    Sigmoid, //> Also known as 'Logistic' function
    ReLU,
    Tanh,
    LeakyReLU, // Avoids the dead neurons of ReLU by using a shallow linear part for negative values.
    Identity,
    Threshold, //> Also known as 'Step' or 'Binary step'
    SoftPlus,
    ArcTan,
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

enum class DType
{
    Float16,
    Float32
};

struct TrainingResultTracker
{
    std::atomic<float> m_epoch_progress = 0;
    std::atomic<uint32_t> m_epochs_finished = 0;
    std::future<uint32_t> m_future;

    mutable std::atomic<bool> m_stop_at_next_epoch = false;
};

inline bool GetBoolFlagFromJson(const nlohmann::json& json_struct, const std::string& param_name, bool default_value)
{
    if (json_struct.contains(param_name) && json_struct[param_name].is_boolean()) {
        return json_struct[param_name].get<bool>();
    }
    return default_value;
}

inline int GetIntFromJson(const nlohmann::json& json_struct, const std::string& param_name, int default_value)
{
    if (json_struct.contains(param_name) && json_struct[param_name].is_number_integer()) {
        return json_struct[param_name].get<int>();
    }
    return default_value;
}

} // namespace macademy

#define ASSERTM(x, msg)                                                                                                                                                                                \
    if (!(x)) {                                                                                                                                                                                        \
        throw std::runtime_error("Assertion failed! " msg);                                                                                                                                            \
    }

#define ASSERT(x) ASSERTM(x, "")