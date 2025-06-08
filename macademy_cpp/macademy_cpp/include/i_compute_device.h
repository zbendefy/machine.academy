#pragma once
#include <vector>
#include <span>
#include <memory>
#include <string>
#include <variant>

#include <i_buffer.h>
#include <common.h>

namespace macademy {
class Network;
struct TrainingSuite;

struct ComputeDeviceInfo
{
    std::string m_backend;
    uint32_t m_device_index;
    std::string m_device_name;
    uint64_t m_total_memory;

    bool operator==(ComputeDeviceInfo const&) const = default;
};

class IComputeDevice
{
  public:
    virtual ~IComputeDevice() {}

    virtual std::unique_ptr<IBuffer> CreateBuffer(size_t size, BufferUsage buffer_usage, const std::string& name) = 0;

    virtual void QueueWriteToBuffer(IBuffer* dst_buffer, std::span<const uint8_t> src, size_t buffer_offset) = 0;
    virtual void QueueReadFromBuffer(IBuffer* src_buffer, std::span<uint8_t> dst, size_t buffer_offset) = 0;
    virtual void QueueFillBuffer(IBuffer* buffer, uint32_t data, size_t offset_bytes, size_t size_bytes) = 0;
    virtual void SubmitQueue() = 0;
    virtual void WaitQueueIdle() = 0;

    virtual void QueueEvaluateLayer(const IBuffer* tensor_buffer, const IBuffer* layer_input_buffer, IBuffer* layer_output_buffer, ActivationFunction activation,
        uint32_t layer_input_count, uint32_t layer_neuron_count) = 0;
    virtual void QueueTrainForwardPass(const IBuffer* tensor_buffer, const IBuffer* prev_activations, IBuffer* activations, IBuffer* zvalues, ActivationFunction activation_function,
        uint32_t layer_neuron_count, uint32_t weights_per_neuron, uint32_t num_training_samples) = 0;
    virtual void QueueTrainBackwardPass(const IBuffer* next_layer_tensor_buffer, const IBuffer* prev_activations_buffer, const IBuffer* layer_activations_buffer, const IBuffer* layer_zvalues_buffer,
        IBuffer* delta_k_vector_buffer_write, const IBuffer* delta_k_vector_buffer_read, IBuffer* current_layer_gradient_buffer, const IBuffer* desiredOutputsBuffer, uint32_t layer_neuron_count, uint32_t weights_per_neuron, ActivationFunction activation_function, uint32_t numTrainingSamples, CostFunction costFunction, uint32_t next_layer_neuron_count) = 0;
    virtual void QueueApplyGradients(IBuffer* tensor_buffer, const IBuffer* gradient_buffer, uint32_t layer_neuron_count, uint32_t weights_per_neuron, float regularization_term_1, float regularization_term_2, float normalized_learning_rate) = 0;

    virtual std::string GetDeviceName() const = 0;
    virtual size_t GetTotalMemory() const = 0;
    virtual bool SupportsWeightFormat(DType format) const = 0;
};
} // namespace macademy