#include "cpu_backend/cpu_compute_backend.h"
#include "network.h"
#include "common.h"
#include "utils.h"
#include "training_suite.h"
#include "hwinfo/hwinfo.h"
#include <execution>
#include <algorithm>

namespace macademy {
namespace {

int GetLayerNeuronCountOffset(int layerId, const uint32_t* layer_config)
{
    int offset = 0;
    for (int i = 0; i < layerId; ++i) {
        offset += layer_config[2 + i * 2]; // neuron count of i-th layer
    }
    return offset;
}

inline float CalculateActivationFunction(ActivationFunction func, float x)
{
    switch (func) {
    case ActivationFunction::Sigmoid:
        return 1.0f / (1.0f + expf(-x));
    case ActivationFunction::ReLU:
        return x < 0.0f ? 0.0f : x;
    case ActivationFunction::Tanh:
        return 2.0f / (1.0f + expf(-2.0f * x)) - 1.0f;
    case ActivationFunction::Identity:
        return x;
    case ActivationFunction::Threshold:
        return x < 0 ? 0 : 1;
    case ActivationFunction::LeakyReLU:
        return x < 0.0f ? (0.01f * x) : x;
    case ActivationFunction::SoftPlus:
        return logf(1 + exp(x));
    case ActivationFunction::ArcTan:
        return atanf(x);
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
    case ActivationFunction::Tanh: {
        const float sigm = CalculateActivationFunction(ActivationFunction::Tanh, x);
        return 1.0f - sigm * sigm;
    }
    case ActivationFunction::Identity:
        return 1.0f;
    case ActivationFunction::Threshold:
        return 0.0f;
    case ActivationFunction::LeakyReLU:
        return x < 0.0f ? 0.01f : 1.0f;
    case ActivationFunction::SoftPlus:
        return 1.0f / (1.0f + expf(-x));
    case ActivationFunction::ArcTan:
        return 1.0f / (x * x + 1);
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

inline float CalculateCostFunctionDelta(CostFunction cost_fnc, ActivationFunction activation_function, float z, float a, float desired_output)
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

std::unique_ptr<IBuffer> CPUComputeDevice::CreateBuffer(size_t size, BufferUsage, const std::string& name)
{
    auto ret = std::make_unique<CPUBuffer>();
    ret->m_data.resize(size);

    return ret;
}

void CPUComputeDevice::QueueWriteToBuffer(IBuffer* dst_buffer, std::span<const uint8_t> src, size_t buffer_offset)
{
    CPUBuffer* cpu_buffer = BufferCast<CPUBuffer>(dst_buffer);

    ASSERT(cpu_buffer->m_data.size() >= buffer_offset + src.size());

    memcpy(cpu_buffer->m_data.data() + buffer_offset, src.data(), src.size());
}

void CPUComputeDevice::QueueReadFromBuffer(IBuffer* src_buffer, std::span<uint8_t> dst, size_t buffer_offset)
{
    CPUBuffer* cpu_buffer = BufferCast<CPUBuffer>(src_buffer);

    ASSERT(cpu_buffer->m_data.size() >= buffer_offset + dst.size());

    memcpy(dst.data(), cpu_buffer->m_data.data() + buffer_offset, dst.size());
}

void CPUComputeDevice::QueueFillBuffer(IBuffer* buffer, uint32_t data, size_t offset_bytes, size_t size_bytes)
{
    CPUBuffer* cpu_buffer = BufferCast<CPUBuffer>(buffer);

    ASSERT(cpu_buffer->m_data.size() >= offset_bytes + size_bytes);

    memset(cpu_buffer->m_data.data() + offset_bytes, data, size_bytes);
}

void CPUComputeDevice::SubmitQueue()
{ /*CPU doesn't queue operations*/
}

void CPUComputeDevice::WaitQueueIdle()
{ /*CPU doesn't queue operations*/
}

void CPUComputeDevice::QueueEvaluateLayerBatched(const IBuffer* weights_buffer, const IBuffer* layer_config_buffer, const IBuffer* layer_input_buffer, IBuffer* layer_output_buffer,
                                                 uint32_t current_layer_id, uint64_t current_layer_weights_offset, uint32_t batch_count, uint32_t layer_neuron_count)
{
    const auto weights_f32 = BufferCast<const CPUBuffer>(weights_buffer)->As<const float>();
    const auto layer_input = BufferCast<const CPUBuffer>(layer_input_buffer)->As<const float>();
    auto layer_output = BufferCast<CPUBuffer>(layer_output_buffer)->As<float>();
    const auto layer_config = BufferCast<const CPUBuffer>(layer_config_buffer)->As<const uint32_t>();

    ASSERT(layer_config[2 + current_layer_id * 2] == layer_neuron_count); // value in the buffer should match

    const uint32_t weights_per_neuron = layer_config[current_layer_id * 2]; // neurons in the prev layer
    const uint32_t activationFunctionId = layer_config[3 + current_layer_id * 2];

    std::for_each_n(std::execution::par_unseq, weights_f32, layer_neuron_count * batch_count, [&](const float& f) {
        const uint32_t neuron_id = &f - weights_f32;

        const uint32_t layer_neuron_id = neuron_id % layer_neuron_count;
        const uint32_t batch_id = neuron_id / layer_neuron_count;

        const float* input = layer_input + batch_id * weights_per_neuron;
        float* output = layer_output + batch_id * layer_neuron_count;

        const uint32_t neuron_data_size = weights_per_neuron + 1; // weights in prev layer + 1 bias

        const float* neuron_weights_biases = weights_f32 + current_layer_weights_offset + layer_neuron_id * neuron_data_size;

        float acc = 0;
        for (int i = 0; i < weights_per_neuron; ++i) {
            acc += neuron_weights_biases[i] * input[i];
        }
        acc += neuron_weights_biases[weights_per_neuron]; // bias

        output[layer_neuron_id] = CalculateActivationFunction(ActivationFunction(activationFunctionId), acc);
    });
}

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

bool CPUComputeDevice::SupportsWeightFormat(NetworkWeightFormat format) const
{
    switch (format) {
    case macademy::NetworkWeightFormat::Float16:
        return true;
    case macademy::NetworkWeightFormat::Float32:
        return true;
    }

    throw std::runtime_error("CPUComputeDevice::SupportsWeightFormat: Invalid NetworkWeightFormat!");
}
void CPUComputeDevice::QueueTrainForwardPass(const IBuffer* weights_buffer, const IBuffer* layer_config_buffer, IBuffer* m_activations_zvalues_buffer, const IBuffer* input_buffer,
                                             uint32_t layer_neuron_count, uint32_t current_layer_id, uint64_t current_layer_weights_offset, uint32_t num_training_samples, uint32_t total_neuron_count)
{
    const auto weights_f32 = BufferCast<const CPUBuffer>(weights_buffer)->As<const float>();
    const auto layer_config = BufferCast<const CPUBuffer>(layer_config_buffer)->As<const uint32_t>();
    auto activations_zvalues = BufferCast<CPUBuffer>(m_activations_zvalues_buffer)->As<float>();
    const auto input_data = BufferCast<const CPUBuffer>(input_buffer)->As<const float>();

    const uint32_t weights_per_neuron = layer_config[current_layer_id * 2]; // neurons in the prev layer
    const uint32_t activationFunctionId = layer_config[3 + current_layer_id * 2];

    // TODOZ parallel for
    for (size_t g_id = 0; g_id < layer_neuron_count * num_training_samples; ++g_id) {
        const uint32_t layer_neuron_id = g_id % layer_neuron_count;
        const uint32_t trainingSampleId = g_id / layer_neuron_count;

        const uint32_t neuron_data_size = weights_per_neuron + 1; // weights in prev layer + 1 bias

        const float* neuron_weights_biases = weights_f32 + current_layer_weights_offset + layer_neuron_id * neuron_data_size;

        const int training_sample_activation_offset = total_neuron_count * trainingSampleId;

        const uint32_t input_layer_neuron_count = layer_config[0];
        const float* prevActivations = current_layer_id == 0 ? (input_data + input_layer_neuron_count * trainingSampleId)
                                                             : (activations_zvalues + (training_sample_activation_offset + GetLayerNeuronCountOffset(current_layer_id - 1, layer_config)));

        // Calculate ZValues for layer
        float acc = 0;
        for (int i = 0; i < weights_per_neuron; ++i) {
            acc += neuron_weights_biases[i] * prevActivations[i];
        }
        acc += neuron_weights_biases[weights_per_neuron]; // bias

        // Store ZValues and the result of the activation function
        const int layer_activation_offset = training_sample_activation_offset + GetLayerNeuronCountOffset(current_layer_id, layer_config);
        const int layer_zvalue_offset = layer_activation_offset + (num_training_samples * total_neuron_count); // zvalues are stored after activations, so shift by the number of total activations
        activations_zvalues[layer_zvalue_offset + layer_neuron_id] = acc;
        activations_zvalues[layer_activation_offset + layer_neuron_id] = CalculateActivationFunction(ActivationFunction(activationFunctionId), acc);
    }
}

void CPUComputeDevice::QueueTrainBackwardPass(const IBuffer* weights_buffer, const IBuffer* layer_config_buffer, const IBuffer* m_activations_zvalues_buffer, const IBuffer* input_buffer,
                                              IBuffer* delta_k_vector_buffer, IBuffer* gradient_buffer, const IBuffer* desiredOutputsBuffer, uint32_t layer_neuron_count, uint32_t current_layer_id,
                                              uint32_t layer_count, uint32_t numTrainingSamples, uint32_t total_neuron_count, CostFunction costFunction, uint32_t largest_layer_neuron_count,
                                              uint64_t layer_weights_offset)
{
    const auto weights_f32 = BufferCast<const CPUBuffer>(weights_buffer)->As<const float>();
    const auto layer_config = BufferCast<const CPUBuffer>(layer_config_buffer)->As<const uint32_t>();
    const auto activations_zvalues = BufferCast<const CPUBuffer>(m_activations_zvalues_buffer)->As<const float>();
    const auto input_data = BufferCast<const CPUBuffer>(input_buffer)->As<const float>();
    auto delta_k_vector = BufferCast<CPUBuffer>(delta_k_vector_buffer)->As<float>();
    auto gradient = BufferCast<CPUBuffer>(gradient_buffer)->As<float>();
    const auto desiredOutputs = BufferCast<const CPUBuffer>(desiredOutputsBuffer)->As<const float>();

    const uint32_t weights_per_neuron = layer_config[current_layer_id * 2]; // neurons in the prev layer
    const uint32_t activationFunctionId = layer_config[3 + current_layer_id * 2];
    const uint32_t deltaKVectorStride = largest_layer_neuron_count; // Table size of delta_k vector

    //__constant const int* layerNeuronCountBegin = config+8;

    for (size_t g_id = 0; g_id < layer_neuron_count * numTrainingSamples; ++g_id) {

        const uint32_t layer_neuron_id = g_id % layer_neuron_count;
        const uint32_t trainingSampleId = g_id / layer_neuron_count;

        const int delta_k_read_offset = deltaKVectorStride * 2 * trainingSampleId + ((current_layer_id % 2) * deltaKVectorStride);
        const int delta_k_write_offset = deltaKVectorStride * 2 * trainingSampleId + (((current_layer_id + 1) % 2) * deltaKVectorStride);

        const int training_sample_activation_offset = total_neuron_count * trainingSampleId;
        const int current_layer_activation_offset = training_sample_activation_offset + GetLayerNeuronCountOffset(current_layer_id, layer_config);
        const int current_layer_z_values_offset = current_layer_activation_offset + (numTrainingSamples * total_neuron_count); // shift by table size

        const uint32_t input_layer_neuron_count = layer_config[0];
        const float* prevActivations = current_layer_id == 0 ? (input_data + input_layer_neuron_count * trainingSampleId)
                                                             : (activations_zvalues + (training_sample_activation_offset + GetLayerNeuronCountOffset(current_layer_id - 1, layer_config)));

        const float zValue = activations_zvalues[current_layer_z_values_offset + layer_neuron_id];

        float delta_k;

        if (current_layer_id == (layer_count - 1)) {
            // Output layer
            const float activation = activations_zvalues[current_layer_activation_offset + layer_neuron_id];
            const float desiredOutput = desiredOutputs[trainingSampleId * layer_neuron_count + layer_neuron_id];
            delta_k = CalculateCostFunctionDelta(costFunction, ActivationFunction(activationFunctionId), zValue, activation, desiredOutput);
        } else {
            // Hidden layer
            delta_k = 0;
            const uint32_t next_layer_weights_offset = layer_weights_offset + (layer_neuron_count * (weights_per_neuron + 1));
            const uint32_t next_layer_weight_offset_for_processed_neuron = next_layer_weights_offset + layer_neuron_id;
            const uint32_t next_layer_neuron_count = layer_config[2 + (current_layer_id + 1) * 2]; // number of neurons
            const uint32_t next_layer_neuron_data_size = layer_neuron_count + 1;                   // weights + bias
            for (uint32_t i = 0; i < next_layer_neuron_count; ++i) {
                delta_k += delta_k_vector[delta_k_read_offset + i] * weights_f32[next_layer_weight_offset_for_processed_neuron + (i * next_layer_neuron_data_size)];
            }
            delta_k *= CalculateActivationFunctionPrime(ActivationFunction(activationFunctionId), zValue);
        }

        const uint32_t gradientBaseOffset = layer_weights_offset + layer_neuron_id * (weights_per_neuron + 1);

        for (int i = 0; i < weights_per_neuron; ++i) {
            float& v = *(gradient + gradientBaseOffset + i);
            v += delta_k * prevActivations[i];
        }
        float& v = *(gradient + gradientBaseOffset + weights_per_neuron);
        v += delta_k; // bias

        if (current_layer_id != 0) {
            delta_k_vector[delta_k_write_offset + layer_neuron_id] = delta_k;
        }
    }
}

void CPUComputeDevice::QueueApplyGradients(IBuffer* weights_buffer, const IBuffer* gradient_buffer, const IBuffer* layer_config_buffer, uint32_t layer_neuron_count, uint32_t current_layer_id,
                                           uint64_t layer_weight_data_offset, float regularization_term_1, float regularization_term_2, float normalized_learning_rate)
{
    auto weights_f32 = BufferCast<CPUBuffer>(weights_buffer)->As<float>();
    const auto gradient = BufferCast<const CPUBuffer>(gradient_buffer)->As<const float>();
    const auto layer_config = BufferCast<const CPUBuffer>(layer_config_buffer)->As<const uint32_t>();

    const uint32_t weights_per_neuron = layer_config[current_layer_id * 2]; // neurons in the prev layer
    const bool applyRegularizationTerm2 = regularization_term_2 != 0.0f;

    const auto layer_weight_data = weights_f32 + layer_weight_data_offset;
    auto layer_gradient_data = gradient + layer_weight_data_offset;
    const auto neuron_data_size = weights_per_neuron + 1;

    // TODO: async
    for (size_t i = 0; i < layer_neuron_count; ++i) {
        auto neuron_weight_bias_data = layer_weight_data + i * neuron_data_size;
        for (size_t j = 0; j < weights_per_neuron; ++j) {
            auto weight_data = neuron_weight_bias_data + j;
            const auto g = *(layer_gradient_data++);
            *weight_data = regularization_term_1 * (*weight_data) - g * normalized_learning_rate;
            if (applyRegularizationTerm2) {
                *weight_data -= regularization_term_2 * sign(*weight_data);
            }
        }
        *(neuron_weight_bias_data + weights_per_neuron) -= *(layer_gradient_data++) * normalized_learning_rate; // bias
    }
}

ComputeDeviceInfo CPUComputeDevice::GetCpuComputeDeviceInfo()
{
    hwinfo::CPU cpu;
    hwinfo::RAM ram;
    return ComputeDeviceInfo{.m_backend = "cpu", .m_device_index = 0, .m_device_name = cpu.getModelName(), .m_total_memory = uint64_t(ram.getTotalSize_Bytes())};
}

} // namespace macademy