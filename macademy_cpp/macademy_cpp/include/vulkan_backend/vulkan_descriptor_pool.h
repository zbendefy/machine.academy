#pragma once

#include <vulkan_backend/vulkan_common.h>

#include <span>
#include <optional>
#include <set>

namespace macademy::vk {
class Device;

class DescriptorPool
{
    VkDevice m_device;
    VkDescriptorPool m_descriptor_pool;

  public:
    DescriptorPool(Device* device, const std::string& name, std::span<VkDescriptorPoolSize> descriptor_counts, uint32_t max_sets);

    ~DescriptorPool() { vkDestroyDescriptorPool(m_device, m_descriptor_pool, nullptr); }

    VkDescriptorPool GetHandle() { return m_descriptor_pool; }
};

} // namespace macademy::vk

