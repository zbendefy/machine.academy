#include "cpu_compute_backend.h"
#include "network.h"
#include "common.h"
#include "utils.h"
#include "training_suite.h"
#include "hwinfo/hwinfo.h"
#include <execution>
#include <algorithm>

namespace macademy {
namespace {

inline float CalculateActivationFunction(ActivationFunction func, float x)
{
    switch (func) {
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
    case ActivationFunction::Sigmoid: {
        const float sigm = CalculateActivationFunction(ActivationFunction::Sigmoid, x);
        return sigm * (1.0f - sigm);
    }
    case ActivationFunction::ReLU:
        return x < 0.0f ? 0.0f : 1.0f;
    }

    throw std::runtime_error("Invalid activation function!");
}

inline float CalculateCostFunctionError(CostFunction cost_fnc, float result, float desired_output)
{
    switch (cost_fnc) {
    case CostFunction::MeanSquared: {
        float v = result - desired_output;
        return 0.5f * v * v;
    }
    case CostFunction::CrossEntropy_Sigmoid: {
        return -desired_output * logf(result) - (1.0f - desired_output) * logf(1.0f - result);
    }
    }
    throw std::runtime_error("Invalid cost function!");
}

inline float CalculateCostFunctionDelta(CostFunction cost_fnc, float z, float a, float desired_output, ActivationFunction activation_function)
{
    switch (cost_fnc) {
    case CostFunction::MeanSquared: {
        return (a - desired_output) * CalculateActivationFunctionPrime(activation_function, z);
    }
    case CostFunction::CrossEntropy_Sigmoid: {
        return a - desired_output;
    }
    }
    throw std::runtime_error("Invalid cost function!");
}

} // namespace

void CPUComputeDevice::Train(const NetworkResourceHandle& network_handle, const TrainingSuite& training_suite, uint32_t trainingDataBegin, uint32_t trainingDataEnd) const
{
    ASSERT(trainingDataBegin < trainingDataEnd);

    auto& network = *network_handle.m_network;
    const auto network_layout = network.GetLayerConfig();

    const auto accumulated_gradient = CalculateAccumulatedGradientForBatch(network_handle, training_suite, trainingDataBegin, trainingDataEnd);

    // Calculate regularization terms based on the training configuration
    float regularizationTerm1 = 1.0f;
    float regularizationTerm2Base = 0.0f;
    if (training_suite.m_regularization == Regularization::L2) {
        regularizationTerm1 = 1.0f - training_suite.m_learning_rate * (training_suite.m_regularization_lambda / (float)training_suite.m_training_data.size());
    } else if (training_suite.m_regularization == Regularization::L1) {
        regularizationTerm2Base = -((training_suite.m_learning_rate * (training_suite.m_regularization_lambda / (float)training_suite.m_training_data.size())));
    }
    const bool applyRegularizationTerm2 = regularizationTerm2Base != 0.0f;

    const float normalized_learning_rate = training_suite.m_learning_rate * (float(trainingDataEnd - trainingDataBegin) / (float)training_suite.m_training_data.size());

    // apply_gradient
    std::for_each(std::execution::par_unseq, network_layout.begin(), network_layout.end(), [&](const LayerConfig& layer_config) {
        const uint32_t layer_id = &layer_config - &network_layout[0];
        const auto layer_weight_data_offset = GetOffsetToLayerWeights(network, layer_id);
        const auto layer_weight_data = network.GetRawWeightData().data() + layer_weight_data_offset;
        auto layer_gradient_data = accumulated_gradient.data() + layer_weight_data_offset;
        const auto weights_per_neuron = GetLayerWeightsPerNeuronCount(network, layer_id);
        const auto neuron_data_size = weights_per_neuron + 1;

        for (size_t i = 0; i < layer_config.m_num_neurons; ++i) {
            auto neuron_weight_bias_data = layer_weight_data + i * neuron_data_size;
            for (size_t j = 0; j < weights_per_neuron; ++j) {
                auto weight_data = neuron_weight_bias_data + j;
                const auto g = *(layer_gradient_data++);
                *weight_data = regularizationTerm1 * (*weight_data) - g * normalized_learning_rate;
                if (applyRegularizationTerm2) {
                    *weight_data -= regularizationTerm2Base * sign(*weight_data);
                }
            }
            *(neuron_weight_bias_data + weights_per_neuron) -= *(layer_gradient_data++) * normalized_learning_rate; // bias
        }

        ASSERT(layer_gradient_data - (accumulated_gradient.data() + layer_weight_data_offset) == (GetOffsetToLayerWeights(network, layer_id + 1) - GetOffsetToLayerWeights(network, layer_id)));
    });
}

std::vector<float> CPUComputeDevice::CalculateAccumulatedGradientForBatch(const NetworkResourceHandle& network_handle, const TrainingSuite& training_suite, uint32_t batch_begin,
                                                                          uint32_t batch_end) const
{
    const Network& network = *network_handle.m_network;

    const uint32_t total_neurons = network.GetNeuronCount();

    std::vector<float> accumulated_gradient;
    accumulated_gradient.resize(network.GetRawWeightData().size(), 0.0f);

    for (int i = batch_begin; i < batch_end; ++i) {
        const auto& training_input = training_suite.m_training_data[i].m_input;
        const auto& desired_output = training_suite.m_training_data[i].m_desired_output;

        std::optional<InterimTrainingData> interim_data = InterimTrainingData{total_neurons};
        EvaluateAndCollectInterimData(network_handle, training_input, interim_data);

        std::vector<float> delta_k_buffer;
        delta_k_buffer.resize(CalculateLargestLayerNeuronCount(network.GetLayerConfig()));

        CalculateOutputLayerGradient(network, training_suite.m_cost_function, accumulated_gradient, delta_k_buffer, *interim_data, training_input, desired_output);

        for (int i = network.GetLayerConfig().size() - 2; i >= 0; --i) {
            CalculateHiddenLayerGradient(network, i, accumulated_gradient, delta_k_buffer, *interim_data, training_input);
        }
    }

    return accumulated_gradient;
}

void CPUComputeDevice::CalculateOutputLayerGradient(const Network& network, CostFunction cost_function, std::span<float> gradient_data, std::span<float> delta_k_vector,
                                                    const InterimTrainingData& interim_data, const std::vector<float>& training_input, const std::vector<float>& desired_output) const
{
    const int last_layer_idx = network.GetLayerConfig().size() - 1;
    const uint32_t last_layer_neuron_count = network.GetLayerConfig()[last_layer_idx].m_num_neurons;
    const uint32_t last_layer_weight_count = last_layer_idx == 0 ? network.GetInputCount() : network.GetLayerConfig()[last_layer_idx - 1].m_num_neurons;
    std::span<const float> prev_activations; // Activations in the previous layer
    if (last_layer_idx == 0) {
        prev_activations = training_input;
    } else {
        const uint32_t prev_layer_neuron_count = last_layer_weight_count;
        const size_t begin = interim_data.m_activations.size() - last_layer_neuron_count - prev_layer_neuron_count;
        const size_t end = interim_data.m_activations.size() - last_layer_neuron_count;
        prev_activations = std::span<const float>(&interim_data.m_activations[begin], &interim_data.m_activations[end]);
    }
    std::span<const float> last_layer_activations =
        std::span<const float>(interim_data.m_activations.end() - last_layer_neuron_count, interim_data.m_activations.end());                                    // Activations in the last layer
    std::span<const float> last_layer_z_values = std::span<const float>(interim_data.m_z_values.end() - last_layer_neuron_count, interim_data.m_z_values.end()); // Z Values in the last layer
    const size_t last_layer_weight_and_bias_count = last_layer_neuron_count * (last_layer_weight_count + 1);
    std::span<float> gradient_last_layer = std::span<float>(gradient_data.end() - last_layer_weight_and_bias_count, gradient_data.end());
    auto activationFunction = network.GetLayerConfig()[last_layer_idx].m_activation;
    size_t g_id = 0;
    for (int i = 0; i < last_layer_neuron_count; i++) {
        const float outputValue = last_layer_activations[i];
        const float delta_k = CalculateCostFunctionDelta(cost_function, last_layer_z_values[i], outputValue, desired_output[i], activationFunction);

        for (int j = 0; j < last_layer_weight_count; j++) {
            gradient_last_layer[g_id++] += delta_k * prev_activations[j];
        }
        gradient_last_layer[g_id++] += delta_k;
        delta_k_vector[i] = delta_k;
    }
}

void CPUComputeDevice::CalculateHiddenLayerGradient(const Network& network, uint32_t layer_id, std::span<float> gradient_data, std::span<float> delta_k_vector, const InterimTrainingData& interim_data,
                                                    const std::vector<float>& training_input) const
{
    const ActivationFunction activation_fnc = network.GetLayerConfig()[layer_id].m_activation;
    uint32_t layer_weight_count = layer_id == 0 ? training_input.size() : network.GetLayerConfig()[layer_id - 1].m_num_neurons;
    uint32_t layer_neuron_count = network.GetLayerConfig()[layer_id].m_num_neurons;
    uint32_t next_layer_neuron_count = network.GetLayerConfig()[layer_id + 1].m_num_neurons;
    std::span<const float> next_layer_weights = std::span<const float>(network.GetRawWeightData().begin() + GetOffsetToLayerWeights(network, layer_id + 1),
                                                                       network.GetRawWeightData().begin() + GetOffsetToLayerWeights(network, layer_id + 2));
    std::span<float> current_layer_gradient_data =
        std::span<float>(gradient_data.begin() + GetOffsetToLayerWeights(network, layer_id), gradient_data.begin() + GetOffsetToLayerWeights(network, layer_id + 1));

    std::span<const float> layer_z_values = std::span<const float>(interim_data.m_z_values.begin() + GetOffsetToLayerNeuronCount(network.GetLayerConfig(), layer_id),
                                                                   interim_data.m_z_values.begin() + GetOffsetToLayerNeuronCount(network.GetLayerConfig(), layer_id + 1));
    std::span<const float> prev_layer_activations = layer_id == 0 ? training_input
                                                                  : std::span<const float>(interim_data.m_activations.begin() + GetOffsetToLayerNeuronCount(network.GetLayerConfig(), layer_id - 1),
                                                                                           interim_data.m_activations.begin() + GetOffsetToLayerNeuronCount(network.GetLayerConfig(), layer_id));

    std::vector<float> newGammak;
    newGammak.resize(layer_neuron_count);

    uint64_t grad_id = 0;
    for (uint32_t i = 0; i < layer_neuron_count; ++i) {
        float deltak = 0;
        const uint32_t next_layer_neuron_data_size = layer_neuron_count + 1; // weights + bias
        for (int k = 0; k < next_layer_neuron_count; ++k) {
            deltak += delta_k_vector[k] * next_layer_weights[k * next_layer_neuron_data_size + i];
        }
        deltak *= CalculateActivationFunctionPrime(activation_fnc, layer_z_values[i]);
        newGammak[i] = deltak;

        for (int j = 0; j < layer_weight_count; ++j) {
            current_layer_gradient_data[grad_id++] += deltak * (prev_layer_activations[j]);
        }
        current_layer_gradient_data[grad_id++] += deltak; // bias
    }

    memcpy(delta_k_vector.data(), newGammak.data(), newGammak.size() * sizeof(float));
}

std::vector<float> CPUComputeDevice::EvaluateAndCollectInterimData(const NetworkResourceHandle& network_handle, std::span<const float> input,
                                                                   std::optional<InterimTrainingData>& output_interim_data) const
{
    Network& network = *network_handle.m_network;

    if (input.size() != network.GetInputCount()) {
        throw std::runtime_error("Invalid input length!");
    }

    auto layer_config = network.GetLayerConfig();
    std::vector<float> layer_args = std::vector<float>(input.begin(), input.end());
    std::vector<float> layer_result{};
    const float* layer_weight_data = network.GetRawWeightData().data();

    uint64_t activation_offset = 0;

    for (size_t i = 0; i < layer_config.size(); ++i) {
        layer_result.clear();
        const uint32_t input_num = uint32_t(layer_args.size());
        const uint32_t output_num = layer_config[i].m_num_neurons;
        ActivationFunction activation_fnc = layer_config[i].m_activation; // If Z values are required,

        layer_result.resize(output_num);

        std::for_each_n(std::execution::par_unseq, network.GetRawWeightData().begin(), output_num, [&](const float& f) {
            const uint32_t neuron_id = &f - &network.GetRawWeightData()[0];
            float acc = 0.0f;
            const float* neuron_weight_data = layer_weight_data + (input_num + 1) * neuron_id; // pointer to the weights of this neuron
            for (uint32_t weight_id = 0; weight_id < input_num; weight_id++) {
                acc += neuron_weight_data[weight_id] * layer_args[weight_id];
            }
            acc += neuron_weight_data[input_num]; // bias
            // TODO: acc may become too large here in case of large networks, handle NaN!
            layer_result[neuron_id] = output_interim_data ? acc : CalculateActivationFunction(activation_fnc, acc);
        });

        if (output_interim_data) {
            memcpy(output_interim_data->m_z_values.data() + activation_offset, layer_result.data(), layer_result.size() * sizeof(float));

            ActivationFunction actual_activation_function = layer_config[i].m_activation;
            // If z value output was required, we had to skip the activation function to be able to provide it (by using passtrough). Here we apply the real activation function
            std::for_each(std::execution::par_unseq, layer_result.begin(), layer_result.end(),
                          [actual_activation_function](float& f) { f = CalculateActivationFunction(actual_activation_function, f); });

            memcpy(output_interim_data->m_activations.data() + activation_offset, layer_result.data(), layer_result.size() * sizeof(float));

            activation_offset += layer_result.size();
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

std::vector<float> CPUComputeDevice::Evaluate(const NetworkResourceHandle& network_handle, std::span<const float> input) const
{
    std::optional<InterimTrainingData> empty{};
    return EvaluateAndCollectInterimData(network_handle, input, empty);
}
} // namespace macademy