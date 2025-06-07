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

void CPUComputeDevice::QueueEvaluateLayer(const IBuffer* tensor_buffer, const IBuffer* layer_input_buffer, IBuffer* layer_output_buffer, ActivationFunction activation,
                                                uint32_t layer_input_count, uint32_t layer_neuron_count)
{
    const auto weights_f32 = BufferCast<const CPUBuffer>(tensor_buffer)->As<const float>();
    const auto layer_input = BufferCast<const CPUBuffer>(layer_input_buffer)->As<const float>();
    auto layer_output = BufferCast<CPUBuffer>(layer_output_buffer)->As<float>();

    const uint32_t weights_per_neuron = layer_input_count; // neurons in the prev layer

    std::for_each_n(std::execution::par_unseq, weights_f32, layer_neuron_count, [&](const float& f) {
        const uint32_t neuron_id = &f - weights_f32;

        const float* input = layer_input + weights_per_neuron;
        float* output = layer_output + layer_neuron_count;

        const uint32_t neuron_data_size = weights_per_neuron + 1; // weights in prev layer + 1 bias

        const float* neuron_weights_biases = weights_f32 + neuron_id * neuron_data_size;

        float acc = 0;
        for (int i = 0; i < weights_per_neuron; ++i) {
            acc += neuron_weights_biases[i] * input[i];
        }
        acc += neuron_weights_biases[weights_per_neuron]; // bias

        output[neuron_id] = CalculateActivationFunction(activation, acc);
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

bool CPUComputeDevice::SupportsWeightFormat(DType format) const
{
    switch (format) {
    case macademy::DType::Float16:
        return true;
    case macademy::DType::Float32:
        return true;
    }

    throw std::runtime_error("CPUComputeDevice::SupportsWeightFormat: Invalid NetworkWeightFormat!");
}

void CPUComputeDevice::QueueTrainForwardPass(const IBuffer* tensor_buffer, const IBuffer* prev_activations_buffer, bool share_prev_activations_among_samples, IBuffer* activations, IBuffer* zvalues, ActivationFunction activation_function,
    uint32_t layer_neuron_count, uint32_t weights_per_neuron, uint32_t num_training_samples)
{
    const auto weights_f32 = BufferCast<const CPUBuffer>(tensor_buffer)->As<const float>();
    auto activations_f32 = BufferCast<CPUBuffer>(activations)->As<float>();
    auto zvalues_f32 = BufferCast<CPUBuffer>(zvalues)->As<float>();
    auto prev_activations_base = BufferCast<const CPUBuffer>(prev_activations_buffer)->As<const float>(); //layer_input
    const uint32_t& prev_layer_neuron_count = weights_per_neuron;

    const uint32_t neuron_data_size = weights_per_neuron + 1; // weights in prev layer + 1 bias
    const uint64_t training_data_stride = layer_neuron_count;
    auto prev_activations = prev_activations_base;

    // TODOZ parallel for
    for (size_t g_id = 0; g_id < layer_neuron_count * num_training_samples; ++g_id) {
        const uint32_t layer_neuron_id = g_id % layer_neuron_count;
        const uint32_t trainingSampleId = g_id / layer_neuron_count;

        const float* neuron_weights_biases = weights_f32 + layer_neuron_id * neuron_data_size;
        const int training_sample_activation_offset = training_data_stride * trainingSampleId;

        if (!share_prev_activations_among_samples)
        {
            prev_activations = prev_activations_base + prev_layer_neuron_count * trainingSampleId;
        }

        // Calculate ZValues for layer
        float acc = 0;
        for (int i = 0; i < weights_per_neuron; ++i) {
            acc += neuron_weights_biases[i] * prev_activations[i];
        }
        acc += neuron_weights_biases[weights_per_neuron]; // bias

        // Store ZValues and the result of the activation function
        const int layer_activation_offset = training_sample_activation_offset;
        const int layer_zvalue_offset = training_sample_activation_offset;
        zvalues_f32[layer_zvalue_offset + layer_neuron_id] = acc;
        activations_f32[layer_activation_offset + layer_neuron_id] = CalculateActivationFunction(activation_function, acc);
    }
}

void CPUComputeDevice::QueueTrainBackwardPass(const IBuffer* next_layer_tensor_buffer, const IBuffer* prev_activations_buffer, bool share_prev_activations_among_samples, const IBuffer* layer_activations_buffer, const IBuffer* layer_zvalues_buffer,
                                              IBuffer* delta_k_vector_buffer_write, const IBuffer* delta_k_vector_buffer_read, IBuffer* current_layer_gradient_buffer, const IBuffer* desiredOutputsBuffer, uint32_t layer_neuron_count, uint32_t weights_per_neuron, ActivationFunction activation_function, uint32_t numTrainingSamples, CostFunction costFunction, uint32_t next_layer_neuron_count)
{
    //Next layer: the subsequent layer of the network towards the output of the whole network
    //Prev layer: the previous layer of the network towards the input of the whole network

    //TODOZ split this into two functions, one for hidden layers and one for the output layer. Or maybe do the full separation
    const bool is_output_layer = !next_layer_tensor_buffer;
    const auto next_layer_weights_f32 = is_output_layer ? nullptr : BufferCast<const CPUBuffer>(next_layer_tensor_buffer)->As<const float>();
    auto prev_activations_base = BufferCast<const CPUBuffer>(prev_activations_buffer)->As<const float>();
    auto layer_activations = BufferCast<const CPUBuffer>(layer_activations_buffer)->As<const float>();
    auto layer_zvalues = BufferCast<const CPUBuffer>(layer_zvalues_buffer)->As<const float>();
    auto delta_k_vector_read = BufferCast<const CPUBuffer>(delta_k_vector_buffer_read)->As<float>();
    auto delta_k_vector_write = BufferCast<CPUBuffer>(delta_k_vector_buffer_write)->As<float>();
    auto current_layer_gradient = BufferCast<CPUBuffer>(current_layer_gradient_buffer)->As<float>();
    const auto desiredOutputs = BufferCast<const CPUBuffer>(desiredOutputsBuffer)->As<const float>();
    const uint32_t& prev_layer_neuron_count = weights_per_neuron;

    auto prev_activations = prev_activations_base;

    for (size_t g_id = 0; g_id < layer_neuron_count * numTrainingSamples; ++g_id) {

        const uint32_t layer_neuron_id = g_id % layer_neuron_count;
        const uint32_t trainingSampleId = g_id / layer_neuron_count;

        const int delta_k_read_offset = next_layer_neuron_count * trainingSampleId;
        const int delta_k_write_offset = layer_neuron_count * trainingSampleId;

        if (!share_prev_activations_among_samples)
        {
            prev_activations = prev_activations_base + prev_layer_neuron_count * trainingSampleId;
        }

        const float zValue = layer_zvalues[layer_neuron_id];

        float delta_k;

        if (is_output_layer) {
            // Output layer
            const float activation = layer_activations[layer_neuron_id];
            const float desiredOutput = desiredOutputs[trainingSampleId * layer_neuron_count + layer_neuron_id];
            delta_k = CalculateCostFunctionDelta(costFunction, activation_function, zValue, activation, desiredOutput);
        } else {
            // Hidden layer
            delta_k = 0;
            const uint32_t next_layer_neuron_data_size = layer_neuron_count + 1;                   // weights + bias
            for (uint32_t i = 0; i < next_layer_neuron_count; ++i) {
                delta_k += delta_k_vector_read[delta_k_read_offset + i] * next_layer_weights_f32[i * next_layer_neuron_data_size];
            }
            delta_k *= CalculateActivationFunctionPrime(activation_function, zValue);
        }

        const uint32_t gradientBaseOffset = layer_neuron_id * (weights_per_neuron + 1);

        for (int i = 0; i < weights_per_neuron; ++i) {
            float& v = *(current_layer_gradient + gradientBaseOffset + i);
            v += delta_k * prev_activations[i];
        }
        float& v = *(current_layer_gradient + gradientBaseOffset + weights_per_neuron);
        v += delta_k; // bias

        //TODOZ: if this is the input layer of the network, this write is unnecessary, as it won't be used. This write can be omitted
        delta_k_vector_write[delta_k_write_offset + layer_neuron_id] = delta_k;
    }
}

void CPUComputeDevice::QueueApplyGradients(IBuffer* tensor_buffer, const IBuffer* gradient_buffer, uint32_t layer_neuron_count, uint32_t weights_per_neuron, float regularization_term_1, float regularization_term_2, float normalized_learning_rate)
{
    auto weights_f32 = BufferCast<CPUBuffer>(tensor_buffer)->As<float>();
    const auto gradient = BufferCast<const CPUBuffer>(gradient_buffer)->As<const float>();

    const bool applyRegularizationTerm2 = regularization_term_2 != 0.0f;

    const auto neuron_data_size = weights_per_neuron + 1;

    // TODO: async
    size_t g_id = 0;
    for (size_t i = 0; i < layer_neuron_count; ++i) {
        auto neuron_weight_bias_data = weights_f32 + i * neuron_data_size;
        for (size_t j = 0; j < weights_per_neuron; ++j) {
            auto weight_data = neuron_weight_bias_data + j;
            const auto g = gradient[g_id++];
            *weight_data = regularization_term_1 * (*weight_data) - g * normalized_learning_rate;
            if (applyRegularizationTerm2) {
                *weight_data -= regularization_term_2 * sign(*weight_data);
            }
        }
        *(neuron_weight_bias_data + weights_per_neuron) -= gradient[g_id++] * normalized_learning_rate; // bias
    }
}

ComputeDeviceInfo CPUComputeDevice::GetCpuComputeDeviceInfo()
{
    hwinfo::CPU cpu;
    hwinfo::RAM ram;
    return ComputeDeviceInfo{.m_backend = "cpu", .m_device_index = 0, .m_device_name = cpu.getModelName(), .m_total_memory = uint64_t(ram.getTotalSize_Bytes())};
}

} // namespace macademy