#pragma once

#include "vulkan_backend/vulkan_common.h"
#include "common.h"

#include <VmaUsage.h>
#include <memory>

namespace macademy::vk {

class Device;

class VulkanBuffer
{
    Device* m_device = nullptr;
    VmaAllocator& m_allocator;
    VkBuffer m_buffer;
    VmaAllocation m_allocation;
    size_t m_size;
    void* m_persistently_mapped_data = nullptr;
    std::string m_name;

  public:
    VulkanBuffer(Device* device, const std::string& name, size_t size, VkBufferUsageFlags usage_flags, VmaMemoryUsage vma_memory_usage, VmaAllocationCreateFlags alloc_create_flags);

    const std::string& GetName() const { return m_name; }

    void UpdateData(const std::span<uint8_t>& data, size_t offset);

    VkBuffer GetHandle() { return m_buffer; }

    size_t GetSize() const { return m_size; }

    void* MapMemory()
    {
        if (m_persistently_mapped_data) {
            return m_persistently_mapped_data;
        }

        void* data;
        vmaMapMemory(m_allocator, m_allocation, &data);
        return data;
    }

    void UnmapMemory()
    {
        if (!m_persistently_mapped_data) {
            vmaUnmapMemory(m_allocator, m_allocation);
        }
    }

    ~VulkanBuffer();
};
} // namespace macademy::vk