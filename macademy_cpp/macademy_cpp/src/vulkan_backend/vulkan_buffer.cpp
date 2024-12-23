#include <vulkan_backend/vulkan_buffer.h>
#include <vulkan_backend/vulkan_device.h>
#include <vulkan_backend/vulkan_instance.h>

namespace macademy::vk {
VulkanBuffer::VulkanBuffer(Device* device, const std::string& name, size_t size, VkBufferUsageFlags usage_flags, VmaMemoryUsage vma_memory_usage, VmaAllocationCreateFlags alloc_create_flags)
    : m_device(device), m_allocator(device->GetVMAAllocator()), m_size(size), m_name(name)
{
    VkBufferCreateInfo bufCreateInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufCreateInfo.size = size;
    bufCreateInfo.usage = usage_flags;

    VmaAllocationCreateInfo allocCreateInfo = {};
    allocCreateInfo.usage = vma_memory_usage;
    allocCreateInfo.flags = alloc_create_flags;

    VmaAllocationInfo allocInfo;
    if (vmaCreateBuffer(m_allocator, &bufCreateInfo, &allocCreateInfo, &m_buffer, &m_allocation, &allocInfo) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create buffer: " + name);
    }

    m_persistently_mapped_data = allocInfo.pMappedData; // nullptr if no persistently_mapped bit was requested

    device->GetInstance()->SetDebugObjectName(device, uint64_t(m_buffer), name.c_str(), VK_OBJECT_TYPE_BUFFER);
}

VulkanBuffer::~VulkanBuffer() { vmaDestroyBuffer(m_allocator, m_buffer, m_allocation); }
} // namespace macademy::vk
