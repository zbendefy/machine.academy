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

    void QueueEvaluateLayerBatched(const IBuffer* weights_buffer, const IBuffer* layer_config_buffer, const IBuffer* layer_input_buffer, IBuffer* layer_output_buffer, uint32_t layer_id,
                                   uint64_t weights_layer_offset, uint32_t batch_count, uint32_t layer_neuron_count) override;

    std::string GetDeviceName() const;
    size_t GetTotalMemory() const;
    bool SupportsWeightFormat(NetworkWeightFormat format) const;

    static ComputeDeviceInfo GetCpuComputeDeviceInfo();
};

} // namespace macademy