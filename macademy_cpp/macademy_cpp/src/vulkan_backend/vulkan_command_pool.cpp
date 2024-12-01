#include "vulkan_backend/vulkan_command_pool.h"

#include "vulkan_backend/vulkan_device.h"
#include "vulkan_backend/vulkan_instance.h"

#include <exception>

namespace macademy::vk {

CommandPool::CommandPool(Device* device, const std::string& name, uint32_t queue_family_index) : m_device(device->GetHandle())
{
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = queue_family_index;

    if (vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_command_pool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create command pool!");
    }

    device->GetInstance()->SetDebugObjectName(device, uint64_t(m_command_pool), name.c_str(), VK_OBJECT_TYPE_COMMAND_POOL);
}

} // namespace macademy::vk