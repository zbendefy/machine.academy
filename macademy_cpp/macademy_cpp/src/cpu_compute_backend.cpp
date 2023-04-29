#include "cpu_compute_backend.h"
#include "network.h"
#include "common.h"
#include "training_suite.h"
#include "hwinfo/hwinfo.h"
#include <execution>
#include <algorithm>

namespace macademy {
namespace {

    inline float CalculateActivationFunction(ActivationFunction func, float x)
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

inline float CalculateActivationFunctionPrime(ActivationFunction func, float x)
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

void CPUComputeDevice::TrainOnMinibatch(const NetworkResourceHandle& network_handle, const TrainingSuite& training_suite)
{ 
    uint32_t batch_begin = 0;
    uint32_t batch_end = 10;

    Network& network = *network_handle.m_network;

    const uint32_t total_neurons = network.GetNeuronCount();

    for (int i = batch_begin; i < batch_end; ++i) {
        const auto& training_input = training_suite.m_training_data[i].m_input;

        std::optional<InterimTrainingData> interim_data = InterimTrainingData{total_neurons};
        EvaluateAndCollectInterimData(network_handle, std::span<float>(training_input.data(), training_data.size()), interim_data);

    }


}

std::vector<float> CPUComputeDevice::EvaluateAndCollectInterimData(const NetworkResourceHandle& network_handle, const std::span<float>& input, std::optional<InterimTrainingData>& output_interim_data) const
{
    Network& network = *network_handle.m_network;

    if (input.size() != network.GetInputCount()) {
        throw std::runtime_error("Invalid input length!");
    }

    auto layer_config = network.GetLayerConfig();
    std::vector<float> layer_args = std::vector<float>(input.begin(), input.end());
    std::vector<float> layer_result{};
    const float* layer_weight_data = network.GetRawWeightData().data();

    uint32_t interim_data_written = 0;

    for (size_t i = 0; i < layer_config.size(); ++i) {
        layer_result.clear();
        const uint32_t input_num = uint32_t(layer_args.size());
        const uint32_t output_num = layer_config[i].m_num_neurons;
        ActivationFunction activation_fnc = output_interim_data ? ActivationFunction::Passtrough : layer_config[i].m_activation; // If Z values are required, 

        layer_result.resize(output_num);

        std::for_each_n(std::execution::par_unseq, network.GetRawWeightData().begin(), output_num, [&network, input_num, &layer_weight_data, &layer_args, &layer_result, activation_fnc](const float& f) {
            const uint32_t neuron_id = &f - &network.GetRawWeightData()[0];
            float acc = 0.0f;
            const float* neuron_weight_data = layer_weight_data + (input_num + 1) * neuron_id; // pointer to the weights of this neuron
            for (uint32_t weight_id = 0; weight_id < input_num; weight_id++) {
                acc += neuron_weight_data[weight_id] * layer_args[weight_id];
            }
            acc += neuron_weight_data[input_num]; // bias
            // TODO: acc may become too large here in case of large networks, handle NaN!
            layer_result[neuron_id] = CalculateActivationFunction(activation_fnc, acc);
        });

        if (output_interim_data)
        {
            memcpy(output_interim_data->m_z_values.data(), layer_result.data(), layer_result.size() * sizeof(float));

            ActivationFunction actual_activation_function = layer_config[i].m_activation;
            //If z value output was required, we had to skip the activation function to be able to provide it (by using passtrough). Here we apply the real activation function
            std::for_each(std::execution::par_unseq, layer_result.begin(), layer_result.end(),
                          [actual_activation_function](float& f) { f = CalculateActivationFunction(actual_activation_function, f); });

            memcpy(output_interim_data->m_activations.data(), layer_result.data(), layer_result.size() * sizeof(float));

            interim_data_written += layer_result.size();
        }

        layer_weight_data += input_num * output_num + output_num; // Advance the pointer to the weights of the layer
        std::swap(layer_args, layer_result);
    }

    ASSERTM(layer_weight_data - network.GetRawWeightData().data() == network.GetRawWeightData().size(), "");
    return layer_args;
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

std::vector<float> CPUComputeDevice::Evaluate(const NetworkResourceHandle& network_handle, const std::span<float>& input) const {
    std::optional<InterimTrainingData> empty{};
    return EvaluateAndCollectInterimData(network_handle, input, empty);
}
} // namespace macademy