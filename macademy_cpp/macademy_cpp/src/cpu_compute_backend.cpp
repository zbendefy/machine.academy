#include "cpu_compute_backend.h"
#include "network.h"
#include "common.h"
#include "training_suite.h"
#include "hwinfo/hwinfo.h"
#include <execution>
#include <algorithm>

namespace macademy {
namespace {
float CalculateActivationFunction(ActivationFunction func, float x)
{
    switch (func) {
    case ActivationFunction::Passtrough:
        return x;
    case ActivationFunction::Sigmoid:
        return 1.0f / (1.0f + std::expf(-x));
    case ActivationFunction::ReLU:
        return x < 0.0f ? 0.0f : x;
    }

    throw std::runtime_error("Invalid activation function!");
}

float CalculateActivationFunctionPrime(ActivationFunction func, float x)
{
    switch (func) {
    case ActivationFunction::Passtrough:
        return 0;
    case ActivationFunction::Sigmoid: {
        const float sigm = CalculateActivationFunction(ActivationFunction::Sigmoid, x);
        return sigm * (1.0f - sigm);
    }
    case ActivationFunction::ReLU:
        return x < 0.0f ? 0.0f : 1.0f;
    }

    throw std::runtime_error("Invalid activation function!");
}
} // namespace

void CPUComputeDevice::Train(const NetworkResourceHandle& network_handle, const TrainingSuite& training_suite) const
{
    Network& network = *network_handle.m_network;

    TrainingResultTracker future;

    if (training_suite.m_epochs < 1) {
        return;
    }

    std::async(std::launch::async, [&training_suite]() {
        for (int currentEpoch = 0; currentEpoch < training_suite.m_epochs; currentEpoch++) {
            // if (stopatnextepoch) return;

            if (training_suite.m_shuffle_training_data) {
                // TODO
            }
        }
    });
}

std::unique_ptr<NetworkResourceHandle> CPUComputeDevice::RegisterNetwork(Network& network) { return std::make_unique<NetworkResourceHandle>(network); }

std::string CPUComputeDevice::GetDeviceName() const
{
    hwinfo::CPU cpu;
    return "CPU device: " + cpu.getModelName();
}

size_t CPUComputeDevice::GetTotalMemory() const
{
    hwinfo::RAM ram;
    return size_t(ram.getTotalSize_Bytes());
}

uint32_t CPUComputeDevice::GetComputeUnits() const
{
    hwinfo::CPU cpu;
    return cpu.getNumLogicalCores();
}

std::vector<float> CPUComputeDevice::Evaluate(const NetworkResourceHandle& network_handle, const std::span<float>& input) const
{
    Network& network = *network_handle.m_network;

    if (input.size() != network.GetInputCount()) {
        throw std::runtime_error("Invalid input length!");
    }

    auto layer_config = network.GetLayerConfig();
    std::vector<float> layer_args = std::vector<float>(input.begin(), input.end());
    std::vector<float> layer_result{};
    const float* layer_weight_data = network.GetRawWeightData().data();
    for (size_t i = 0; i < layer_config.size(); ++i) {
        layer_result.clear();
        const uint32_t input_num = uint32_t(layer_args.size());
        const uint32_t output_num = layer_config[i].m_num_neurons;
        const ActivationFunction activation = layer_config[i].m_activation;

        layer_result.resize(output_num);
        
        std::for_each_n(std::execution::par_unseq, network.GetRawWeightData().begin(), output_num, [&network, input_num, &layer_weight_data, &layer_args, &layer_result, activation](const float& f) {
            const uint32_t neuron_id = &f - &network.GetRawWeightData()[0];
            float acc = 0.0f;
            const float* neuron_weight_data = layer_weight_data + (input_num + 1) * neuron_id; //pointer to the weights of this neuron
            for (uint32_t weight_id = 0; weight_id < input_num; weight_id++) {
                acc += neuron_weight_data[weight_id] * layer_args[weight_id];
            }
            acc += neuron_weight_data[input_num]; //bias
            // TODO: acc may become too large here in case of large networks, handle NaN!
            layer_result[neuron_id] = CalculateActivationFunction(activation, acc);
            
        });

        layer_weight_data += input_num * output_num + output_num; //Advance the pointer to the weights of the layer
        std::swap(layer_args, layer_result);
    }

    ASSERTM(layer_weight_data - network.GetRawWeightData().data() == network.GetRawWeightData().size(), "");
    return layer_args;
}
} // namespace macademy