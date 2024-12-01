#pragma once

#include <vulkan/vulkan.h>
#include <vulkan_backend/vulkan_common.h>

#include <optional>
#include <set>
#include <string>

namespace macademy::vk {
class Device;

class CommandPool
{
    VkDevice m_device{};
    VkCommandPool m_command_pool{};

  public:
    CommandPool(Device* device, const std::string& name, uint32_t queue_family_index);

    ~CommandPool() { vkDestroyCommandPool(m_device, m_command_pool, nullptr); }

    VkCommandPool GetHandle() { return m_command_pool; }
};

} // namespace macademy::vk

