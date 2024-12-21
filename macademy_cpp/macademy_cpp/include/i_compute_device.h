#pragma once
#include <vector>
#include <span>
#include <memory>
#include <string>
#include <variant>

namespace macademy {
class Network;
class TrainingSuite;

struct UniformDistribution 
{
    float range;
};

class IBuffer
{
public:
    virtual ~IBuffer() {}
};

template <typename T> T* BufferCast(IBuffer* i_buf) 
{ 
    T* ret = dynamic_cast<T*>(i_buf);
    if (!ret) {
        throw std::exception("Invalid buffer cast!");
    }

    return ret;
}

template <typename T> const T* BufferCast(const IBuffer* i_buf)
{
    T* ret = dynamic_cast<const T*>(i_buf);
    if (!ret) {
        throw std::exception("Invalid buffer cast!");
    }

    return ret;
}

class IComputeDevice
{
  public:
    virtual ~IComputeDevice() {}

    virtual std::unique_ptr<IBuffer> CreateBuffer(size_t size, bool is_read_only_from_device, std::span<uint8_t> initial_data) = 0;

    virtual void QueueWriteToBuffer(IBuffer* buffer, std::span<uint8_t> data, size_t offset) = 0;
    virtual void QueueReadFromBuffer(IBuffer* buffer, std::span<uint8_t> data, size_t offset) = 0;
    virtual void QueueFillBuffer(IBuffer* buffer, uint32_t data, size_t offset_bytes, size_t size_bytes) = 0;
    virtual void SubmitQueue() = 0;
    virtual void WaitQueueIdle() = 0;

    virtual void QueueEvaluateLayerBatched(const IBuffer* weights_buffer, const IBuffer* layer_config_buffer, const IBuffer* const layer_input_buffer, IBuffer* layer_output_buffer,
                                           uint32_t layer_id, uint64_t weights_layer_offset, uint32_t batch_count) = 0;

};
} // namespace macademy