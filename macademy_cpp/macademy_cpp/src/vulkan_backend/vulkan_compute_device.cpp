#include "vulkan_backend/vulkan_compute_device.h"
#include "vulkan_backend/vulkan_buffer.h"
#include "vulkan_backend/shader_specialization.h"
#include "vulkan_backend/vulkan_buffer.h"
#include "network.h"
#include "common.h"
#include "utils.h"
#include "training_suite.h"

#include <fstream>
#include <sstream>

namespace macademy {

VkCommandBuffer& VulkanComputeDevice::GetCommandBuffer()
{
    if (m_current_command_buffer == VK_NULL_HANDLE) {
        m_current_command_buffer = m_device->CreateCommandBuffer();
    }

    return m_current_command_buffer;
}

VulkanComputeDevice::VulkanComputeDevice(vk::Device* device) : m_device(device)
{
    vk::ShaderSpecializationMap shader_specialization;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &uboLayoutBinding;

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor set layout!");
    }
    vk::ComputePipelineDescriptor kernel_calc_single_layer_pipeline_desc{};
    kernel_calc_single_layer_pipeline_desc.m_compute_shader = 0;
    kernel_calc_single_layer_pipeline_desc.m_compute_shader_entry_function = "kernel_calc_single_layer";
    kernel_calc_single_layer_pipeline_desc.m_descriptor_set_layout = ;
    kernel_calc_single_layer_pipeline_desc.m_push_constant_size = 0;

    m_kernel_calc_single_layer = std::make_unique<vk::ComputePipeline>(m_device, "kernel_calc_single_layer", kernel_calc_single_layer_pipeline_desc, shader_specialization);
}

std::unique_ptr<IBuffer> VulkanComputeDevice::CreateBuffer(size_t size, BufferUsage buffer_usage, const std::string& name)
{
    auto ret = std::make_unique<vk::VulkanBuffer>(m_device, name.c_str(), size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);

    return ret;
}

void VulkanComputeDevice::QueueWriteToBuffer(IBuffer* dst_buffer, std::span<const uint8_t> src, size_t buffer_offset) {}

void VulkanComputeDevice::QueueReadFromBuffer(IBuffer* src_buffer, std::span<uint8_t> dst, size_t buffer_offset) {}

void VulkanComputeDevice::QueueFillBuffer(IBuffer* buffer, uint32_t data, size_t offset_bytes, size_t size_bytes)
{
    auto vk_buffer = BufferCast<vk::VulkanBuffer>(buffer);
    vkCmdFillBuffer(GetCommandBuffer(), vk_buffer->GetHandle(), VkDeviceSize(offset_bytes), VkDeviceSize(size_bytes), data);
}

void VulkanComputeDevice::SubmitQueue()
{
    if (m_current_command_buffer != VK_NULL_HANDLE) {
        VkSubmitInfo submit_info{};
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &m_current_command_buffer;

        vkQueueSubmit(m_device->GetComputeQueue(), 1, &submit_info, VK_NULL_HANDLE);
    }
}

void VulkanComputeDevice::WaitQueueIdle()
{
    if (m_current_command_buffer != VK_NULL_HANDLE) {
        vkQueueWaitIdle(m_device->GetComputeQueue());

        // Invalidate command buffer
        vkResetCommandBuffer(m_current_command_buffer, VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT); //??? revise
        m_current_command_buffer = VK_NULL_HANDLE;
    }
}

void VulkanComputeDevice::QueueEvaluateLayerBatched(const IBuffer* weights_buffer, const IBuffer* layer_config_buffer, const IBuffer* layer_input_buffer, IBuffer* layer_output_buffer,
                                                    uint32_t layer_id, uint64_t weights_layer_offset, uint32_t batch_count, uint32_t layer_neuron_count)
{
    const auto weights_buffer_cl = BufferCast<const vk::VulkanBuffer>(weights_buffer);
    const auto layer_config_buffer_cl = BufferCast<const vk::VulkanBuffer>(layer_config_buffer);
    const auto layer_input_buffer_cl = BufferCast<const vk::VulkanBuffer>(layer_input_buffer);
    auto layer_output_buffer_cl = BufferCast<vk::VulkanBuffer>(layer_output_buffer);
}

std::string VulkanComputeDevice::GetDeviceName() const { return "Vulkan Device: " + m_device->GetName(); }

size_t VulkanComputeDevice::GetTotalMemory() const { return 0; }

bool VulkanComputeDevice::SupportsWeightFormat(NetworkWeightFormat format) const
{
    switch (format) {
    case macademy::NetworkWeightFormat::Float16:
        return m_is_float16_supported;
    case macademy::NetworkWeightFormat::Float32:
        return true;
    }

    throw std::runtime_error("VulkanComputeDevice::SupportsWeightFormat: Invalid NetworkWeightFormat!");
}

} // namespace macademy