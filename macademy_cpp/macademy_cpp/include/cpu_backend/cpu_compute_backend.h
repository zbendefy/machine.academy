#pragma once

#include "i_compute_device.h"

#include <optional>

namespace macademy {

    class CPUBuffer : public IBuffer
    {
      public:
        std::vector<uint8_t> m_data;

        template <typename T>
        T* As() { return reinterpret_cast<T*>(m_data.data()); }

        template <typename T>
        const T* As() const { return reinterpret_cast<const T*>(m_data.data()); }
    };
    
    class CPUComputeDevice : public IComputeDevice
    {
    public:
        std::unique_ptr<IBuffer> CreateBuffer(size_t size, bool is_read_only_from_device, std::span<uint8_t> initial_data) = 0;

        void QueueWriteToBuffer(IBuffer* buffer, std::span<uint8_t> data, size_t offset) override;
        void QueueReadFromBuffer(IBuffer* buffer, std::span<uint8_t> data, size_t offset) override;
        void QueueFillBuffer(IBuffer* buffer, uint32_t data, size_t offset, size_t size) override;
        void SubmitQueue() override;
        void WaitQueueIdle() override;

        void QueueEvaluateLayerBatched(const IBuffer* weights_buffer, const IBuffer* layer_config_buffer, const IBuffer* const layer_input_buffer, IBuffer* layer_output_buffer,
                                       uint32_t layer_id, uint64_t weights_layer_offset, uint32_t batch_count) override;

    };

}