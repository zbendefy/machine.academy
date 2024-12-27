#pragma once

#include "i_compute_device.h"
#include "vulkan_common.h"
#include "vulkan_backend/vulkan_device.h"
#include "vulkan_backend/vulkan_instance.h"
#include "vulkan_backend/vulkan_compute_pipeline.h"

#include <optional>
#include <map>

namespace macademy {

struct KernelResources
{
    KernelResources(VkDevice device) : m_device(device) {}

    ~KernelResources()
    {
        if (m_descriptor_set_layout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(m_device, m_descriptor_set_layout, nullptr);
        }
        if (m_descriptor_pool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(m_device, m_descriptor_pool, nullptr);
        }
    }

    VkDescriptorSet GetDescriptorSet(const std::vector<const vk::VulkanBuffer*>& storage_buffers);
    void FreeDescriptorSets();

    VkDevice m_device;
    VkDescriptorSetLayout m_descriptor_set_layout = VK_NULL_HANDLE;
    VkDescriptorPool m_descriptor_pool = VK_NULL_HANDLE;
    mutable std::unique_ptr<vk::ComputePipeline> m_pipeline;

  private:
    std::map<std::vector<const vk::VulkanBuffer*>, VkDescriptorSet> m_descriptor_sets;
};

class VulkanComputeDevice : public IComputeDevice
{
#define VK_CONSTANTS_HOST
#include "vulkan_backend/shaders/constants.h"

    struct MemoryReadback
    {
        std::unique_ptr<vk::Device::LoaderStagingBuffer> m_host_buffer;
        std::span<uint8_t> m_dst;
    };

    PushConstantData m_push_constant_data{};

    std::unique_ptr<vk::Instance> m_instance = nullptr;
    std::unique_ptr<vk::Device> m_device = nullptr;

    std::vector<MemoryReadback> m_memory_reads;

    std::unique_ptr<KernelResources> m_kernel_calc_single_layer;

    enum class BufferSynchronizationEvent
    {
        TransferWrite = 1 << 0,
        ComputeShaderWrite = 1 << 1
    };

    enum class SynchronizationAction
    {
        ComputeShaderRead = 1 << 0,
        TransferRead = 1 << 1
    };

    std::map<const vk::VulkanBuffer*, BufferSynchronizationEvent> m_dirty_buffers;

    // mutable std::unique_ptr<KernelEval> m_kernel_calc_single_layer;
    // mutable std::unique_ptr<KernelTrainingForwardPass> m_kernel_train_forward_pass;
    // mutable std::unique_ptr<KernelTrainingBackwardPass> m_kernel_train_backward_pass;
    // mutable std::unique_ptr<KernelTrainingApplyGradient> m_kernel_train_apply_gradient;

    uint32_t m_kernel_calc_single_layer_ideal_workgroup_size = 64;
    uint32_t m_kernel_training_ideal_workgroup_size = 16;
    uint32_t m_kernel_training_apply_gradient_ideal_workgroup_size = 64;
    bool m_is_float16_supported = false;

    VkCommandBuffer m_current_command_buffer = VK_NULL_HANDLE;

    VkCommandBuffer& GetCommandBuffer();

    void SynchronizeBuffers(VkCommandBuffer command_buffer, SynchronizationAction action, std::span<const vk::VulkanBuffer*> buffers);

  public:
    VulkanComputeDevice(const ComputeDeviceInfo& device);
    ~VulkanComputeDevice();

    std::unique_ptr<IBuffer> CreateBuffer(size_t size, BufferUsage buffer_usage, const std::string& name);

    void QueueWriteToBuffer(IBuffer* dst_buffer, std::span<const uint8_t> src, size_t buffer_offset) override;
    void QueueReadFromBuffer(IBuffer* src_buffer, std::span<uint8_t> dst, size_t buffer_offset) override;
    void QueueFillBuffer(IBuffer* buffer, uint32_t data, size_t offset, size_t size) override;
    void SubmitQueue() override;
    void WaitQueueIdle() override;

    void QueueEvaluateLayerBatched(const IBuffer* weights_buffer, const IBuffer* layer_config_buffer, const IBuffer* layer_input_buffer, IBuffer* layer_output_buffer, uint32_t layer_id,
                                   uint64_t weights_layer_offset, uint32_t batch_count, uint32_t layer_neuron_count) override;

    static std::vector<VkPhysicalDevice> GetDeviceList();

    std::string GetDeviceName() const override;

    size_t GetTotalMemory() const override;

    bool SupportsWeightFormat(NetworkWeightFormat format) const override;

    static std::vector<ComputeDeviceInfo> GetVulkanComputeDeviceInfo();
};
} // namespace macademy