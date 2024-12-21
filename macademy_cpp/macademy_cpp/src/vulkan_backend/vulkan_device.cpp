#include <vulkan_backend/vulkan_device.h>
#include <vulkan_backend/vulkan_instance.h>
#include <vulkan_backend/vulkan_buffer.h>
#include <vulkan_backend/vulkan_command_pool.h>

#include <optional>

namespace macademy::vk {

std::optional<uint32_t> FindComputeQueueFamily(VkPhysicalDevice device)
{
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

    uint32_t i = 0;
    for (const auto& queueFamily : queueFamilies) {
        if (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT) {
            return i;
        }

        i++;
    }

    return {};
}

Device::Device(Instance* instance, VkPhysicalDevice physical_device, bool enable_validation_layer) : m_instance(instance), m_physical_device(physical_device)
{
    auto compute_queue_index = FindComputeQueueFamily(m_physical_device);

    if (!compute_queue_index) {
        throw std::runtime_error("Could not find a compute capable queue!");
    }

    std::vector<const char*> device_extensions{};

    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo computeQueueCreateInfo{};
    computeQueueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    computeQueueCreateInfo.queueFamilyIndex = compute_queue_index.value();
    computeQueueCreateInfo.queueCount = 1;
    computeQueueCreateInfo.pQueuePriorities = &queuePriority;

    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    queueCreateInfos.push_back(computeQueueCreateInfo);

    vkGetPhysicalDeviceProperties(physical_device, &m_device_props);
    vkGetPhysicalDeviceFeatures(physical_device, &m_device_features);
    vkGetPhysicalDeviceMemoryProperties(physical_device, &m_memory_props);

    VkPhysicalDeviceFeatures selected_device_features{};

    VkPhysicalDeviceTimelineSemaphoreFeatures timelineSemaphoreFeatures{};
    timelineSemaphoreFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
    timelineSemaphoreFeatures.timelineSemaphore = VK_TRUE;

    VkPhysicalDeviceFeatures2 physicalDeviceFeatures2{};
    physicalDeviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    physicalDeviceFeatures2.features = selected_device_features;
    physicalDeviceFeatures2.pNext = &timelineSemaphoreFeatures;

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.queueCreateInfoCount = queueCreateInfos.size();
    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.pEnabledFeatures = nullptr; // Comes from VkPhysicalDeviceFeatures2
    createInfo.enabledExtensionCount = static_cast<uint32_t>(device_extensions.size());
    createInfo.ppEnabledExtensionNames = device_extensions.data();
    if (enable_validation_layer) {
        createInfo.enabledLayerCount = 1;
        createInfo.ppEnabledLayerNames = &ValidationLayerExtensionStr;
    } else {
        createInfo.enabledLayerCount = 0;
    }
    createInfo.pNext = &physicalDeviceFeatures2;

    if (vkCreateDevice(m_physical_device, &createInfo, nullptr, &m_device) != VK_SUCCESS) {
        throw std::runtime_error("failed to create logical device!");
    }

    vkGetDeviceQueue(m_device, compute_queue_index.value(), 0, &m_compute_queue);

    {
        // initialize the memory allocator
        VmaAllocatorCreateInfo allocatorInfo = {};
        allocatorInfo.physicalDevice = m_physical_device;
        allocatorInfo.device = m_device;
        allocatorInfo.instance = m_instance->GetHandle();
        allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_3;
        vmaCreateAllocator(&allocatorInfo, &m_vma);
    }

    m_command_pool = std::make_unique<CommandPool>(this, "command_pool", compute_queue_index.value());
}

VkCommandBuffer Device::CreateCommandBuffer()
{
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = m_command_pool->GetHandle();
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer command_buffer;
    if (vkAllocateCommandBuffers(m_device, &allocInfo, &command_buffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate command buffers!");
    }

    m_instance->SetDebugObjectName(this, uint64_t(command_buffer), "Command buffer", VK_OBJECT_TYPE_COMMAND_BUFFER);

    return command_buffer;
}

void Device::RunOneTimeComandBuffer(std::function<void(VkCommandBuffer&)>&& commands)
{
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = m_command_pool->GetHandle();
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(m_device, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    commands(commandBuffer);

    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(m_compute_queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(m_compute_queue);

    vkFreeCommandBuffers(m_device, m_command_pool->GetHandle(), 1, &commandBuffer);
}

void Device::RecycleLoaderBuffer(LoaderStagingBuffer& loader_buffer)
{
    auto it = std::find_if(m_loader_staging_buffers.begin(), m_loader_staging_buffers.end(),
                           [&loader_buffer](const std::pair<std::unique_ptr<Buffer>, bool>& buf) { return buf.first.get() == loader_buffer.m_staging_buffer; });

    ASSERT(it != m_loader_staging_buffers.end());
    it->second = true;
}

std::unique_ptr<Device::LoaderStagingBuffer> Device::GetLoaderStagingBuffer(size_t size)
{
    auto ret = std::make_unique<LoaderStagingBuffer>();
    ret->m_device = this;

    auto it = std::find_if(m_loader_staging_buffers.begin(), m_loader_staging_buffers.end(),
                           [size](const std::pair<std::unique_ptr<Buffer>, bool>& buf) { return buf.second && buf.first->GetSize() >= size; });

    if (it != m_loader_staging_buffers.end()) {
        it->second = false;
        ret->m_staging_buffer = it->first.get();
    } else {
        auto new_buffer = std::make_unique<Buffer>(this, "loader_staging_buffer_" + std::to_string(m_loader_staging_buffers.size()), size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO,
                                                   VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT);
        ret->m_staging_buffer = new_buffer.get();
        m_loader_staging_buffers.emplace_back(std::make_pair(std::move(new_buffer), false));
    }

    return ret;
}

void Device::ClearLoadingBuffers() { m_loader_staging_buffers.clear(); }

Device::~Device()
{
    vkDeviceWaitIdle(m_device);
    m_command_pool.reset();
    vmaDestroyAllocator(m_vma);
    vkDestroyDevice(m_device, nullptr);
}

Device::LoaderStagingBuffer::~LoaderStagingBuffer() { m_device->RecycleLoaderBuffer(*this); }

} // namespace macademy::vk
