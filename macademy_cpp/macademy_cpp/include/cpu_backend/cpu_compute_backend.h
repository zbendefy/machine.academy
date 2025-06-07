#pragma once

#include "i_compute_device.h"

#include <optional>

namespace macademy {

class CPUBuffer : public IBuffer
{
  public:
    std::vector<uint8_t> m_data;

    size_t GetSize() const override { return m_data.size(); }

    template <typename T> T* As() { return reinterpret_cast<T*>(m_data.data()); }

    template <typename T> const T* As() const { return reinterpret_cast<const T*>(m_data.data()); }
};

class CPUComputeDevice : public IComputeDevice
{
  public:
    std::unique_ptr<IBuffer> CreateBuffer(size_t size, BufferUsage buffer_usage, const std::string& name);

    void QueueWriteToBuffer(IBuffer* dst_buffer, std::span<const uint8_t> src, size_t buffer_offset) override;
    void QueueReadFromBuffer(IBuffer* src_buffer, std::span<uint8_t> dst, size_t buffer_offset) override;
    void QueueFillBuffer(IBuffer* buffer, uint32_t data, size_t offset, size_t size) override;
    void SubmitQueue() override;
    void WaitQueueIdle() override;

    void QueueEvaluateLayer(const IBuffer* tensor_buffer, const IBuffer* layer_input_buffer, IBuffer* layer_output_buffer, ActivationFunction activation,
        uint32_t layer_input_count, uint32_t layer_neuron_count) override;
    void QueueTrainForwardPass(const IBuffer* tensor_buffer, const IBuffer* prev_activations, bool share_prev_activations_among_samples, IBuffer* activations, IBuffer* zvalues, ActivationFunction activation_function,
        uint32_t layer_neuron_count, uint32_t weights_per_neuron, uint32_t num_training_samples) override;
    void QueueTrainBackwardPass(const IBuffer* next_layer_tensor_buffer, const IBuffer* prev_activations_buffer, bool share_prev_activations_among_samples, const IBuffer* layer_activations_buffer, const IBuffer* layer_zvalues_buffer,
        IBuffer* delta_k_vector_buffer_write, const IBuffer* delta_k_vector_buffer_read, IBuffer* current_layer_gradient_buffer, const IBuffer* desiredOutputsBuffer, uint32_t layer_neuron_count, uint32_t weights_per_neuron, ActivationFunction activation_function, uint32_t numTrainingSamples, CostFunction costFunction, uint32_t next_layer_neuron_count) override;
    void QueueApplyGradients(IBuffer* tensor_buffer, const IBuffer* gradient_buffer, uint32_t layer_neuron_count, uint32_t weights_per_neuron, float regularization_term_1, float regularization_term_2, float normalized_learning_rate) override;

    std::string GetDeviceName() const;
    size_t GetTotalMemory() const;
    bool SupportsWeightFormat(DType format) const;

    static ComputeDeviceInfo GetCpuComputeDeviceInfo();
};

} // namespace macademy