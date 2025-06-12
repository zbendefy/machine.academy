#pragma once

#include "vulkan_backend/vulkan_common.h"
#include "common.h"
#include "vulkan_buffer.h"
#include "vulkan_backend/vulkan_compute_pipeline.h"
#include <map>

namespace macademy::vk {

class Device;

class ComputeKernel
{
  public:
    ComputeKernel(vk::Device* device, const std::string& name, uint32_t storage_buffer_count, uint32_t push_constant_size, uint32_t max_descriptor_sets, const vk::SpirvBinary& spirv_binary,
                  const vk::ShaderSpecializationMap& shader_specialization);

    ~ComputeKernel();

    void FreeDescriptorSets();
    void Bind(VkCommandBuffer command_buffer, const std::vector<const vk::VulkanBuffer*>& buffers, std::span<const uint8_t> push_constant_data);
    void Dispatch(VkCommandBuffer command_buffer, uint32_t threadgroup_count_x, uint32_t threadgroup_count_y, uint32_t threadgroup_count_z);

  private:
    VkDescriptorSet GetDescriptorSet(const std::vector<const vk::VulkanBuffer*>& storage_buffers);

    vk::Device* m_device;
    VkDescriptorSetLayout m_descriptor_set_layout = VK_NULL_HANDLE;
    VkDescriptorPool m_descriptor_pool = VK_NULL_HANDLE;
    mutable std::unique_ptr<vk::ComputePipeline> m_pipeline;

  private:
    std::map<std::vector<const vk::VulkanBuffer*>, VkDescriptorSet> m_descriptor_sets;
};

} // namespace macademy::vk