#include <vulkan_backend/vulkan_descriptor_pool.h>
#include <vulkan_backend/vulkan_device.h>
#include <vulkan_backend/vulkan_instance.h>

namespace macademy::vk {
DescriptorPool::DescriptorPool(Device* device, const std::string& name, std::span<VkDescriptorPoolSize> descriptor_counts, uint32_t max_sets) : m_device(device->GetHandle())
{
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = descriptor_counts.size();
    poolInfo.pPoolSizes = descriptor_counts.data();
    poolInfo.maxSets = max_sets;

    if (vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_descriptor_pool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor pool!");
    }

    device->GetInstance()->SetDebugObjectName(device, uint64_t(m_descriptor_pool), name.c_str(), VK_OBJECT_TYPE_DESCRIPTOR_POOL);
}
} // namespace macademy::vk
