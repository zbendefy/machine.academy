#pragma once

#include <vulkan/vulkan.h>
#include <vulkan_backend/vulkan_common.h>
#include <VmaUsage.h>

#include <optional>
#include <set>
#include <span>
#include <memory>
#include <functional>

namespace macademy::vk {
class Instance;
class Texture;
class CommandPool;
class VulkanBuffer;

class Device
{
  public:
    struct LoaderStagingBuffer
    {
        Device* m_device = nullptr;
        VulkanBuffer* m_staging_buffer = nullptr;

        ~LoaderStagingBuffer();
    };

    Device(Instance* instance, VkPhysicalDevice physical_device, bool enable_validation_layer);

    VkQueue GetComputeQueue() { return m_compute_queue; }

    Instance* GetInstance() { return m_instance; }

    VkPhysicalDevice GetPhysicalDeviceHandle() { return m_physical_device; }

    VkDevice GetHandle() { return m_device; }

    CommandPool& GetCommandPool() { return *m_command_pool; }

    VmaAllocator& GetVMAAllocator() { return m_vma; }

    const VkPhysicalDeviceMemoryProperties2& GetMemoryProps() const { return m_memory_props; }

    const VkPhysicalDeviceProperties2& GetDeviceProps() const { return m_device_props; }

    const VkPhysicalDeviceFeatures2& GetDeviceFeatures() const { return m_device_features; }

    const VkPhysicalDeviceShaderAtomicFloatFeaturesEXT& GetDeviceAtomicFloatFeatures() const { return m_device_atomic_float_features; }

    void RunOneTimeComandBuffer(std::function<void(VkCommandBuffer&)>&& commands);

    VkCommandBuffer CreateCommandBuffer();

    std::unique_ptr<Device::LoaderStagingBuffer> GetLoaderStagingBuffer(size_t size);

    ~Device();

    std::string GetName() { return m_device_props.properties.deviceName; }

  private:
    void RecycleLoaderBuffer(LoaderStagingBuffer& loader_buffer);

    Instance* m_instance;
    VmaAllocator m_vma;

    VkPhysicalDevice m_physical_device;
    VkDevice m_device;

    VkPhysicalDeviceMemoryProperties2 m_memory_props{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
    VkPhysicalDeviceProperties2 m_device_props{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    VkPhysicalDeviceFeatures2 m_device_features{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT m_device_atomic_float_features{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};

    VkQueue m_compute_queue;

    std::unique_ptr<CommandPool> m_command_pool;

    std::vector<std::pair<std::unique_ptr<VulkanBuffer>, bool>> m_loader_staging_buffers;
};

} // namespace macademy::vk
