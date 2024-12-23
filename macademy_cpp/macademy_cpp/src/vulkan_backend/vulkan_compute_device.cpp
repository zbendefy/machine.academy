#include "vulkan_backend/vulkan_compute_device.h"
#include "vulkan_backend/vulkan_buffer.h"
#include "vulkan_backend/shader_specialization.h"
#include "vulkan_backend/vulkan_buffer.h"
#include "network.h"
#include "common.h"
#include "utils.h"
#include "training_suite.h"

#include <fstream>
#include <array>
#include <sstream>

namespace macademy {

std::vector<ComputeDeviceInfo> VulkanComputeDevice::GetVulkanComputeDeviceInfo()
{
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "macademy";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "foxglove3_macademy";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    VkInstance instance;

    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create vulkan instance!");
    }

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

    std::vector<VkPhysicalDevice> physical_devices;
    physical_devices.resize(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, physical_devices.data());

    uint32_t idx = 0;

    std::vector<ComputeDeviceInfo> ret;

    for (auto& it : physical_devices) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(it, &props);

        VkPhysicalDeviceMemoryProperties memoryProperties;
        vkGetPhysicalDeviceMemoryProperties(it, &memoryProperties);
        VkDeviceSize totalDeviceLocalMemory = 0;
        for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
            if (memoryProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
                totalDeviceLocalMemory += memoryProperties.memoryHeaps[memoryProperties.memoryTypes[i].heapIndex].size;
            }
        }

        ret.push_back(ComputeDeviceInfo{.m_backend = "vulkan", .m_device_index = idx++, .m_device_name = std::string(props.deviceName), .m_total_memory = uint64_t(totalDeviceLocalMemory)});
    }

    vkDestroyInstance(instance, nullptr);

    return ret;
}

struct push_constant_data
{
    uint32_t layer_id;
    uint32_t layer_count;
    uint32_t weights_layer_offset;
    uint32_t numTrainingSamples;
    uint32_t totalActivationCount;
    uint32_t costFunctionId;
    uint32_t largest_layer_neuron_count;
    uint32_t layer_weights_offset;
    float regularization_term_1;
    float regularization_term_2;
    float normalized_learning_rate;
};

VkCommandBuffer& VulkanComputeDevice::GetCommandBuffer()
{
    if (m_current_command_buffer == VK_NULL_HANDLE) {
        m_current_command_buffer = m_device->CreateCommandBuffer();
    }

    return m_current_command_buffer;
}

VulkanComputeDevice::VulkanComputeDevice(const ComputeDeviceInfo& device_info)
{
    m_instance = std::make_unique<vk::Instance>(false, true);

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(m_instance->GetHandle(), &deviceCount, nullptr);
    std::vector<VkPhysicalDevice> physical_devices;
    physical_devices.resize(deviceCount);
    vkEnumeratePhysicalDevices(m_instance->GetHandle(), &deviceCount, physical_devices.data());

    m_device = std::make_unique<vk::Device>(m_instance.get(), physical_devices[device_info.m_device_index], true);

    vk::ShaderSpecializationMap shader_specialization;

    std::array<VkDescriptorSetLayoutBinding, 7> bindings;

    for (int i = 0; i < bindings.size(); ++i) {
        bindings[i] = VkDescriptorSetLayoutBinding{};
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = bindings.size();
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(m_device->GetHandle(), &layoutInfo, nullptr, &m_descriptor_set_layout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor set layout!");
    }

    vk::SpirvBinary shader_spirv; //todo

    vk::ComputePipelineDescriptor kernel_calc_single_layer_pipeline_desc{};
    kernel_calc_single_layer_pipeline_desc.m_compute_shader = shader_spirv;
    kernel_calc_single_layer_pipeline_desc.m_compute_shader_entry_function = "kernel_calc_single_layer";
    kernel_calc_single_layer_pipeline_desc.m_descriptor_set_layout = m_descriptor_set_layout;
    kernel_calc_single_layer_pipeline_desc.m_push_constant_size = sizeof(push_constant_data);

    m_kernel_calc_single_layer = std::make_unique<vk::ComputePipeline>(m_device.get(), "kernel_calc_single_layer", kernel_calc_single_layer_pipeline_desc, shader_specialization);
}

inline VulkanComputeDevice::~VulkanComputeDevice() { vkDestroyDescriptorSetLayout(m_device->GetHandle(), m_descriptor_set_layout, nullptr); }

std::unique_ptr<IBuffer> VulkanComputeDevice::CreateBuffer(size_t size, BufferUsage buffer_usage, const std::string& name)
{
    auto ret = std::make_unique<vk::VulkanBuffer>(m_device.get(), name, size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);

    return ret;
}

void VulkanComputeDevice::QueueWriteToBuffer(IBuffer* dst_buffer, std::span<const uint8_t> src, size_t buffer_offset) 
{
    auto vk_buffer = BufferCast<vk::VulkanBuffer>(dst_buffer);
    auto staging_buffer = m_device->GetLoaderStagingBuffer(src.size());

    auto dst_memory = staging_buffer->m_staging_buffer->MapMemory();
    ASSERT(dst_memory); //loader staging buffers should be host_visible, and therefore mappable!
    memcpy(dst_memory, src.data(), src.size_bytes());
    staging_buffer->m_staging_buffer->UnmapMemory();

    VkBufferCopy copy_region{.srcOffset = 0, .dstOffset = buffer_offset, .size = src.size()};
    vkCmdCopyBuffer(GetCommandBuffer(), staging_buffer->m_staging_buffer->GetHandle(), vk_buffer->GetHandle(), 1, &copy_region);
}

void VulkanComputeDevice::QueueReadFromBuffer(IBuffer* src_buffer, std::span<uint8_t> dst, size_t buffer_offset)
{
    auto vk_buffer = BufferCast<vk::VulkanBuffer>(src_buffer);
    auto staging_buffer = m_device->GetLoaderStagingBuffer(dst.size());

    VkBufferCopy copy_region{.srcOffset = 0, .dstOffset = buffer_offset, .size = dst.size()};
    vkCmdCopyBuffer(GetCommandBuffer(), vk_buffer->GetHandle(), staging_buffer->m_staging_buffer->GetHandle(), 1, &copy_region);

    auto src_memory = staging_buffer->m_staging_buffer->MapMemory();
    ASSERT(src_memory); // loader staging buffers should be host_visible, and therefore mappable!
    memcpy(dst.data(), src_memory, dst.size_bytes());
    staging_buffer->m_staging_buffer->UnmapMemory();
}

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

std::vector<VkPhysicalDevice> VulkanComputeDevice::GetDeviceList() { 
    return std::vector<VkPhysicalDevice>(); 
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