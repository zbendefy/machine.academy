#pragma once

#include "i_compute_device.h"
#include "vulkan_common.h"
#include "vulkan_backend/vulkan_device.h"
#include "vulkan_backend/vulkan_instance.h"

#include <optional>
#include <map>
#include <nlohmann/json.hpp>

namespace macademy {

namespace vk {
class ComputeKernel;
}

class VulkanComputeDevice : public IComputeDevice
{
  private:
    struct MemoryReadback
    {
        std::unique_ptr<vk::Device::LoaderStagingBuffer> m_host_buffer;
        std::span<uint8_t> m_dst;
    };

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

    std::unique_ptr<vk::Instance> m_instance = nullptr;
    std::unique_ptr<vk::Device> m_device = nullptr;

    std::unique_ptr<vk::ComputeKernel> m_kernel_calc_single_layer;
    std::unique_ptr<vk::ComputeKernel> m_kernel_train_forward_pass;
    std::unique_ptr<vk::ComputeKernel> m_kernel_train_backward_pass;
    std::unique_ptr<vk::ComputeKernel> m_kernel_train_apply_gradient;

    std::vector<MemoryReadback> m_memory_reads;

    std::map<const vk::VulkanBuffer*, BufferSynchronizationEvent> m_dirty_buffers;
    std::vector<std::unique_ptr<vk::Device::LoaderStagingBuffer>> m_staging_buffers;

    uint32_t m_kernel_calc_single_layer_ideal_workgroup_size = 64;
    uint32_t m_kernel_training_ideal_workgroup_size_x = 8;
    uint32_t m_kernel_training_ideal_workgroup_size_y = 8;
    uint32_t m_kernel_training_apply_gradient_ideal_workgroup_size = 64;
    bool m_is_float16_supported = false;
    bool m_hw_atomic_add_support = false;

    VkCommandBuffer m_current_command_buffer = VK_NULL_HANDLE;

    VkCommandBuffer& GetCommandBuffer();

    void SynchronizeBuffers(VkCommandBuffer command_buffer, SynchronizationAction action, std::span<const vk::VulkanBuffer*> buffers);

  public:
    VulkanComputeDevice(const ComputeDeviceInfo& device, const nlohmann::json& device_config);
    ~VulkanComputeDevice();

    std::unique_ptr<IBuffer> CreateBuffer(size_t size, BufferUsage buffer_usage, const std::string& name);

    void QueueWriteToBuffer(IBuffer* dst_buffer, std::span<const uint8_t> src, size_t buffer_offset) override;
    void QueueReadFromBuffer(IBuffer* src_buffer, std::span<uint8_t> dst, size_t buffer_offset) override;
    void QueueFillBuffer(IBuffer* buffer, uint32_t data, size_t offset, size_t size) override;
    void SubmitQueue() override;
    void WaitQueueIdle() override;

    void QueueEvaluateLayer(const IBuffer* tensor_buffer, const IBuffer* layer_input_buffer, IBuffer* layer_output_buffer, ActivationFunction activation_function, uint32_t layer_input_count,
                            uint32_t layer_neuron_count) override;
    void QueueTrainForwardPass(const IBuffer* tensor_buffer, const IBuffer* prev_activations, IBuffer* activations, IBuffer* zvalues, ActivationFunction activation_function,
                               uint32_t layer_neuron_count, uint32_t weights_per_neuron, uint32_t num_training_samples) override;
    void QueueTrainBackwardPass(bool is_output_layer, const IBuffer* next_layer_data_buffer, const IBuffer* prev_activations_buffer, const IBuffer* layer_activations_buffer,
                                const IBuffer* layer_zvalues_buffer, IBuffer* delta_k_vector_buffer_write, const IBuffer* delta_k_vector_buffer_read, IBuffer* current_layer_gradient_buffer,
                                uint32_t layer_neuron_count, uint32_t weights_per_neuron, ActivationFunction activation_function, uint32_t num_training_samples, CostFunction costFunction,
                                uint32_t next_layer_neuron_count) override;
    void QueueApplyGradients(IBuffer* tensor_buffer, const IBuffer* gradient_buffer, uint32_t layer_neuron_count, uint32_t weights_per_neuron, float regularization_term_1, float regularization_term_2,
                             float normalized_learning_rate) override;

    std::string GetDeviceName() const override;

    size_t GetTotalMemory() const override;

    bool SupportsWeightFormat(DType format) const override;

    static std::vector<ComputeDeviceInfo> GetVulkanComputeDeviceInfo();
};
} // namespace macademy