#pragma once

#include "i_compute_device.h"
#include "vulkan_common.h"
#include "vulkan_backend/vulkan_device.h"
#include "vulkan_backend/vulkan_instance.h"
#include "vulkan_backend/vulkan_compute_pipeline.h"

#include <optional>

namespace macademy {

class VulkanComputeDevice : public IComputeDevice
{
    vk::Instance* m_instance;
    vk::Device* m_device;

    mutable std::unique_ptr<vk::ComputePipeline> m_kernel_calc_single_layer;

    // mutable std::unique_ptr<KernelEval> m_kernel_calc_single_layer;
    // mutable std::unique_ptr<KernelTrainingForwardPass> m_kernel_train_forward_pass;
    // mutable std::unique_ptr<KernelTrainingBackwardPass> m_kernel_train_backward_pass;
    // mutable std::unique_ptr<KernelTrainingApplyGradient> m_kernel_train_apply_gradient;

    uint32_t m_kernel_calc_single_layer_ideal_workgroup_size = 64;
    uint32_t m_kernel_training_ideal_workgroup_size = 16;
    uint32_t m_kernel_training_apply_gradient_ideal_workgroup_size = 64;
    bool m_is_float16_supported = false;

    VkCommandBuffer m_current_command_buffer;

    VkCommandBuffer& GetCommandBuffer();

  public:
    VulkanComputeDevice(vk::Device* device);

    std::unique_ptr<IBuffer> CreateBuffer(size_t size, BufferUsage buffer_usage, const std::string& name) = 0;

    void QueueWriteToBuffer(IBuffer* dst_buffer, std::span<const uint8_t> src, size_t buffer_offset) override;
    void QueueReadFromBuffer(IBuffer* src_buffer, std::span<uint8_t> dst, size_t buffer_offset) override;
    void QueueFillBuffer(IBuffer* buffer, uint32_t data, size_t offset, size_t size) override;
    void SubmitQueue() override;
    void WaitQueueIdle() override;

    void QueueEvaluateLayerBatched(const IBuffer* weights_buffer, const IBuffer* layer_config_buffer, const IBuffer* layer_input_buffer, IBuffer* layer_output_buffer, uint32_t layer_id,
                                   uint64_t weights_layer_offset, uint32_t batch_count, uint32_t layer_neuron_count) override;

    std::string GetDeviceName() const override;

    size_t GetTotalMemory() const override;

    bool SupportsWeightFormat(NetworkWeightFormat format) const override;
};
} // namespace macademy