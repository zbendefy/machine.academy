#pragma once
#include <vector>
#include <span>
#include <memory>
#include <string>
#include <variant>

#include <i_buffer.h>

namespace macademy {
class Network;
class TrainingSuite;

struct ComputeDeviceInfo
{
    std::string m_backend;
    uint32_t m_device_index;
    std::string m_device_name;
    uint64_t m_total_memory;
};

enum class NetworkWeightFormat
{
    Float16,
    Float32
};

template <typename T> std::span<const uint8_t> ToReadOnlyUi8Span(const T& container)
{
    return std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(container.data()), container.size() * sizeof(typename T::value_type));
}
template <typename T> std::span<const uint8_t> ToReadOnlyUi8Span(std::span<const T> container)
{
    return std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(container.data()), container.size_bytes());
}

template <typename T> std::span<uint8_t> ToWriteableUi8Span(T& container)
{
    return std::span<uint8_t>(reinterpret_cast<uint8_t*>(container.data()), container.size() * sizeof(typename T::value_type));
}
template <typename T> std::span<uint8_t> ToWriteableUi8Span(std::span<T> container) { return std::span<uint8_t>(reinterpret_cast<uint8_t*>(container.data()), container.size_bytes()); }

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

    virtual void QueueEvaluateLayerBatched(const IBuffer* weights_buffer, const IBuffer* layer_config_buffer, const IBuffer* layer_input_buffer, IBuffer* layer_output_buffer, uint32_t layer_id,
                                           uint64_t weights_layer_offset, uint32_t batch_count, uint32_t layer_neuron_count) = 0;
    virtual void QueueTrainForwardPass(const IBuffer* weights_buffer, const IBuffer* layer_config_buffer, const IBuffer* m_activations_zvalues_buffer, const IBuffer* input_buffer, uint32_t output_num,
                                       uint32_t layer_id, uint64_t weights_layer_offset, uint32_t num_training_samples, uint32_t total_neuron_count) = 0;

    virtual std::string GetDeviceName() const = 0;
    virtual size_t GetTotalMemory() const = 0;
    virtual bool SupportsWeightFormat(NetworkWeightFormat format) const = 0;
};
} // namespace macademy