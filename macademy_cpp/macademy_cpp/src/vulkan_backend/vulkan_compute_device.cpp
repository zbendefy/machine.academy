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

#define DEBUG_RENDERDOC

#ifdef DEBUG_RENDERDOC

#include "renderdoc_app.h"

RENDERDOC_API_1_6_0* rdoc_api = NULL;

#endif

namespace macademy {

#include "vulkan_backend/shaders/kernel_calc_single_layer.glsl.h"

VkDescriptorSet KernelResources::GetDescriptorSet(const std::vector<const vk::VulkanBuffer*>& storage_buffers)
{
    auto it = m_descriptor_sets.find(storage_buffers);
    if (it == m_descriptor_sets.end()) {

        // Allocate a descriptor set from the pool, and write the buffer handles into it, then return it

        VkDescriptorSetAllocateInfo allocInfo = {};
        allocInfo.pNext = nullptr;
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = m_descriptor_pool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &m_descriptor_set_layout;

        VkDescriptorSet descriptor_set;
        vkAllocateDescriptorSets(m_device, &allocInfo, &descriptor_set);

        thread_local std::vector<VkDescriptorBufferInfo> buffer_infos;
        buffer_infos.resize(storage_buffers.size());
        for (int i = 0; i < int(storage_buffers.size()); ++i) {
            buffer_infos[i].buffer = storage_buffers[i]->GetHandle();
            buffer_infos[i].offset = 0;
            buffer_infos[i].range = storage_buffers[i]->GetSize();
        }

        thread_local std::vector<VkWriteDescriptorSet> descriptor_writes;
        descriptor_writes.resize(buffer_infos.size());

        for (int i = 0; i < storage_buffers.size(); ++i) {
            descriptor_writes[i] = VkWriteDescriptorSet{};
            descriptor_writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptor_writes[i].pNext = nullptr;
            descriptor_writes[i].dstBinding = i;
            descriptor_writes[i].dstSet = descriptor_set;
            descriptor_writes[i].descriptorCount = 1;
            descriptor_writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptor_writes[i].pBufferInfo = &buffer_infos[i];
        }

        vkUpdateDescriptorSets(m_device, descriptor_writes.size(), descriptor_writes.data(), 0, nullptr);

        m_descriptor_sets.emplace(storage_buffers, descriptor_set);

        return descriptor_set;
    } else {
        return it->second;
    }
}

void KernelResources::FreeDescriptorSets()
{ 
    //Its more efficient to reset the pool, than freeing descriptor sets individually.
    vkResetDescriptorPool(m_device, m_descriptor_pool, 0);

    m_descriptor_sets.clear();
}

namespace {

size_t ExtendGlobalWorkSize(size_t desiredGlobalSize, size_t localSize)
{
    return ((desiredGlobalSize % localSize) == 0) ? desiredGlobalSize : (desiredGlobalSize + (localSize - (desiredGlobalSize % localSize)));
}

vk::SpirvBinary base64_decode_spirv(const std::string& in)
{
    // from: https://stackoverflow.com/questions/180947/base64-decode-snippet-in-c
    vk::SpirvBinary out;

    std::vector<int> T(256, -1);
    for (int i = 0; i < 64; i++)
        T["ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"[i]] = i;

    int val = 0, valb = -8;
    for (uint8_t c : in) {
        if (T[c] == -1)
            break;
        val = (val << 6) + T[c];
        valb += 6;
        if (valb >= 0) {
            out.push_back(char((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return out;
}
} // namespace

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

VkCommandBuffer& VulkanComputeDevice::GetCommandBuffer()
{
    if (m_current_command_buffer == VK_NULL_HANDLE) {
#ifdef DEBUG_RENDERDOC
        if (rdoc_api) {
            rdoc_api->StartFrameCapture(NULL, NULL);
        }
#endif
        m_current_command_buffer = m_device->CreateCommandBuffer();

        VkCommandBufferBeginInfo cmd_buffer_begin_info{};
        cmd_buffer_begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        vkBeginCommandBuffer(m_current_command_buffer, &cmd_buffer_begin_info);
    }

    return m_current_command_buffer;
}

VulkanComputeDevice::VulkanComputeDevice(const ComputeDeviceInfo& device_info)
{
#ifdef DEBUG_RENDERDOC
    if (HMODULE mod = GetModuleHandleA("renderdoc.dll")) {
        pRENDERDOC_GetAPI RENDERDOC_GetAPI = (pRENDERDOC_GetAPI)GetProcAddress(mod, "RENDERDOC_GetAPI");
        int ret = RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_1_2, (void**)&rdoc_api);
        ASSERT(ret == 1);
    }
#endif

    m_instance = std::make_unique<vk::Instance>(false, true);

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(m_instance->GetHandle(), &deviceCount, nullptr);
    std::vector<VkPhysicalDevice> physical_devices;
    physical_devices.resize(deviceCount);
    vkEnumeratePhysicalDevices(m_instance->GetHandle(), &deviceCount, physical_devices.data());

    m_device = std::make_unique<vk::Device>(m_instance.get(), physical_devices[device_info.m_device_index], true);

    {
        m_kernel_calc_single_layer = std::make_unique<KernelResources>(m_device->GetHandle());

        vk::ShaderSpecializationMap shader_specialization;

        constexpr int used_storage_buffers = 4;
        constexpr int max_simultaneous_descriptor_sets = 2;

        std::array<VkDescriptorSetLayoutBinding, used_storage_buffers> bindings;

        for (int i = 0; i < bindings.size(); ++i) {
            bindings[i] = VkDescriptorSetLayoutBinding{};
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = bindings.size();
        layoutInfo.pBindings = bindings.data();
        layoutInfo.flags = 0;

        if (vkCreateDescriptorSetLayout(m_device->GetHandle(), &layoutInfo, nullptr, &m_kernel_calc_single_layer->m_descriptor_set_layout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor set layout!");
        }

        vk::SpirvBinary shader_spirv = base64_decode_spirv(vulkan_kernel_source_kernel_calc_single_layer_glsl_b64);
        ASSERT(shader_spirv.size() % 4 == 0);

        vk::ComputePipelineDescriptor kernel_calc_single_layer_pipeline_desc{};
        kernel_calc_single_layer_pipeline_desc.m_compute_shader = shader_spirv;
        kernel_calc_single_layer_pipeline_desc.m_compute_shader_entry_function = "main";
        kernel_calc_single_layer_pipeline_desc.m_descriptor_set_layout = m_kernel_calc_single_layer->m_descriptor_set_layout;
        kernel_calc_single_layer_pipeline_desc.m_push_constant_size = sizeof(PushConstantData);

        m_kernel_calc_single_layer->m_pipeline = std::make_unique<vk::ComputePipeline>(m_device.get(), "kernel_calc_single_layer", kernel_calc_single_layer_pipeline_desc, shader_specialization);

        std::array<VkDescriptorPoolSize, 1> sizes = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, used_storage_buffers * max_simultaneous_descriptor_sets}};

        VkDescriptorPoolCreateInfo pool_info = {};
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.flags = 0;
        pool_info.maxSets = max_simultaneous_descriptor_sets; // there should be at most only 2 different descriptor sets
        pool_info.poolSizeCount = (uint32_t)sizes.size();
        pool_info.pPoolSizes = sizes.data();

        vkCreateDescriptorPool(m_device->GetHandle(), &pool_info, nullptr, &m_kernel_calc_single_layer->m_descriptor_pool);
    }
}

inline VulkanComputeDevice::~VulkanComputeDevice() { 
    printf("alma");
}

std::unique_ptr<IBuffer> VulkanComputeDevice::CreateBuffer(size_t size, BufferUsage buffer_usage, const std::string& name)
{
    VkBufferUsageFlags buffer_usage_flags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

    auto ret = std::make_unique<vk::VulkanBuffer>(m_device.get(), name, size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                  VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);

    return ret;
}

void VulkanComputeDevice::QueueWriteToBuffer(IBuffer* dst_buffer, std::span<const uint8_t> src, size_t buffer_offset)
{
    auto vk_buffer = BufferCast<vk::VulkanBuffer>(dst_buffer);
    auto staging_buffer = m_device->GetLoaderStagingBuffer(src.size());

    auto dst_memory = staging_buffer->m_staging_buffer->MapMemory();
    ASSERT(dst_memory); // loader staging buffers should be host_visible, and therefore mappable!
    memcpy(dst_memory, src.data(), src.size_bytes());
    staging_buffer->m_staging_buffer->UnmapMemory();

    VkBufferCopy copy_region{.srcOffset = 0, .dstOffset = buffer_offset, .size = src.size_bytes()};
    vkCmdCopyBuffer(GetCommandBuffer(), staging_buffer->m_staging_buffer->GetHandle(), vk_buffer->GetHandle(), 1, &copy_region);

    m_dirty_buffers.emplace(vk_buffer, BufferSynchronizationEvent::TransferWrite);
}

void VulkanComputeDevice::QueueReadFromBuffer(IBuffer* src_buffer, std::span<uint8_t> dst, size_t buffer_offset)
{
    auto vk_buffer = BufferCast<vk::VulkanBuffer>(src_buffer);
    auto staging_buffer = m_device->GetLoaderStagingBuffer(dst.size());
    auto command_buffer = GetCommandBuffer();

    std::array<const vk::VulkanBuffer*, 1> alma{{vk_buffer}};
    SynchronizeBuffers(command_buffer, SynchronizationAction::TransferRead, std::span<const vk::VulkanBuffer*>(alma.begin(), alma.end()));

    VkBufferCopy copy_region{.srcOffset = 0, .dstOffset = buffer_offset, .size = dst.size_bytes()};
    vkCmdCopyBuffer(command_buffer, vk_buffer->GetHandle(), staging_buffer->m_staging_buffer->GetHandle(), 1, &copy_region);

    m_memory_reads.emplace_back();
    m_memory_reads.back().m_host_buffer = std::move(staging_buffer);
    m_memory_reads.back().m_dst = dst;
}

void VulkanComputeDevice::QueueFillBuffer(IBuffer* buffer, uint32_t data, size_t offset_bytes, size_t size_bytes)
{
    auto vk_buffer = BufferCast<vk::VulkanBuffer>(buffer);
    vkCmdFillBuffer(GetCommandBuffer(), vk_buffer->GetHandle(), VkDeviceSize(offset_bytes), VkDeviceSize(size_bytes), data);

    m_dirty_buffers.emplace(vk_buffer, BufferSynchronizationEvent::TransferWrite);
}

void VulkanComputeDevice::SubmitQueue()
{
    if (m_current_command_buffer != VK_NULL_HANDLE) {

        vkEndCommandBuffer(m_current_command_buffer);

        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
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
        vkResetCommandBuffer(m_current_command_buffer, VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT); //??? TODO revise command buffer reuse
        m_current_command_buffer = VK_NULL_HANDLE;

        for (auto& it : m_memory_reads) {
            auto src_memory = it.m_host_buffer->m_staging_buffer->MapMemory();
            ASSERT(src_memory); // loader staging buffers should be host_visible, and therefore mappable!
            memcpy(it.m_dst.data(), src_memory, it.m_dst.size_bytes());
            it.m_host_buffer->m_staging_buffer->UnmapMemory();
        }

        m_memory_reads.clear();

        m_device->ClearLoadingBuffers();

#ifdef DEBUG_RENDERDOC
        if (rdoc_api) {
            rdoc_api->EndFrameCapture(NULL, NULL);
        }
#endif
    }
}

void VulkanComputeDevice::SynchronizeBuffers(VkCommandBuffer command_buffer, SynchronizationAction action, std::span<const vk::VulkanBuffer*> buffers)
{
    thread_local std::vector<VkBufferMemoryBarrier> buffer_memory_barriers;

    const auto Synchronize = [this, &buffers, command_buffer](BufferSynchronizationEvent buffer_event, VkAccessFlags src_access_mask, VkAccessFlags dst_access_mask,
                                                              VkPipelineStageFlagBits dstStageMask) {
        for (int i = 0; i < int(buffers.size()); ++i) {

            auto it = m_dirty_buffers.find(buffers[i]);

            if (it != m_dirty_buffers.end() && (uint32_t(it->second) & uint32_t(buffer_event)) != 0) {
                auto& bufferMemoryBarrier = buffer_memory_barriers.emplace_back(VkBufferMemoryBarrier{});
                bufferMemoryBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                bufferMemoryBarrier.srcAccessMask = src_access_mask; // The copy operation writes to the buffer
                bufferMemoryBarrier.dstAccessMask = dst_access_mask; // The compute shader will read from the buffer
                bufferMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                bufferMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                bufferMemoryBarrier.buffer = buffers[i]->GetHandle();
                bufferMemoryBarrier.offset = 0;
                bufferMemoryBarrier.size = VK_WHOLE_SIZE; // Whole buffer
            }
        }

        if (!buffer_memory_barriers.empty()) {
            VkPipelineStageFlagBits srcStageMask = VkPipelineStageFlagBits(0);

            if (buffer_event == BufferSynchronizationEvent::TransferWrite) {
                srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
            } else if (buffer_event == BufferSynchronizationEvent::ComputeShaderWrite) {
                srcStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
            }

            vkCmdPipelineBarrier(command_buffer,
                                 srcStageMask,                                                 // Source stage: Transfer (copy) is the source stage
                                 dstStageMask,                                                 // Destination stage: Compute shader is the destination stage
                                 0,                                                            // No additional flags
                                 0, nullptr,                                                   // No memory barriers
                                 buffer_memory_barriers.size(), buffer_memory_barriers.data(), // Buffer memory barrier to ensure the copy is done before compute shader reads the buffer
                                 0, nullptr                                                    // No image barriers
            );
        }

        buffer_memory_barriers.clear();
    };

    if (action == SynchronizationAction::ComputeShaderRead) {
        Synchronize(BufferSynchronizationEvent::TransferWrite, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    } else if (action == SynchronizationAction::TransferRead) {
        Synchronize(BufferSynchronizationEvent::TransferWrite, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    }

    for (int i = 0; i < int(buffers.size()); ++i) {

        auto it = m_dirty_buffers.find(buffers[i]);

        if (it != m_dirty_buffers.end() && (uint32_t(it->second) & uint32_t(BufferSynchronizationEvent::ComputeShaderWrite)) != 0) {
            auto& bufferMemoryBarrier = buffer_memory_barriers.emplace_back(VkBufferMemoryBarrier{});
            bufferMemoryBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            bufferMemoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;  // The compute shader writes to the buffer
            bufferMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT; // The copy operation will read from the buffer
            bufferMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bufferMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bufferMemoryBarrier.buffer = buffers[i]->GetHandle();
            bufferMemoryBarrier.offset = 0;
            bufferMemoryBarrier.size = VK_WHOLE_SIZE; // Whole buffer
        }
    }

    if (!buffer_memory_barriers.empty()) {
        vkCmdPipelineBarrier(command_buffer,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,                         // Source stage: Transfer (copy) is the source stage
                             VK_PIPELINE_STAGE_TRANSFER_BIT,                               // Destination stage: Compute shader is the destination stage
                             0,                                                            // No additional flags
                             0, nullptr,                                                   // No memory barriers
                             buffer_memory_barriers.size(), buffer_memory_barriers.data(), // Buffer memory barrier to ensure the copy is done before compute shader reads the buffer
                             0, nullptr                                                    // No image barriers
        );
    }

    buffer_memory_barriers.clear();

    for (int i = 0; i < int(buffers.size()); ++i) {
        m_dirty_buffers.erase(buffers[i]);
    }
}

void VulkanComputeDevice::QueueEvaluateLayerBatched(const IBuffer* weights_buffer, const IBuffer* layer_config_buffer, const IBuffer* layer_input_buffer, IBuffer* layer_output_buffer,
                                                    uint32_t layer_id, uint64_t weights_layer_offset, uint32_t batch_count, uint32_t layer_neuron_count)
{
    const auto weights_buffer_vk = BufferCast<const vk::VulkanBuffer>(weights_buffer);
    const auto layer_config_buffer_vk = BufferCast<const vk::VulkanBuffer>(layer_config_buffer);
    const auto layer_input_buffer_vk = BufferCast<const vk::VulkanBuffer>(layer_input_buffer);
    auto layer_output_buffer_vk = BufferCast<vk::VulkanBuffer>(layer_output_buffer);

    thread_local std::vector<const vk::VulkanBuffer*> buffers;

    buffers.resize(4);
    buffers[0] = BufferCast<const vk::VulkanBuffer>(weights_buffer);
    buffers[1] = BufferCast<const vk::VulkanBuffer>(layer_config_buffer);
    buffers[2] = BufferCast<const vk::VulkanBuffer>(layer_input_buffer);
    buffers[3] = BufferCast<vk::VulkanBuffer>(layer_output_buffer);

    auto command_buffer = GetCommandBuffer();
    auto descriptor_set = m_kernel_calc_single_layer->GetDescriptorSet(buffers);

    SynchronizeBuffers(command_buffer, SynchronizationAction::ComputeShaderRead, std::span<const vk::VulkanBuffer*>(buffers.begin(), buffers.end()));

    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_kernel_calc_single_layer->m_pipeline->GetPipelineLayoutHandle(), 0, 1, &descriptor_set, 0, 0);

    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_kernel_calc_single_layer->m_pipeline->GetHandle());

    m_push_constant_data.layer_id = layer_id;
    m_push_constant_data.weights_layer_offset = weights_layer_offset;
    vkCmdPushConstants(command_buffer, m_kernel_calc_single_layer->m_pipeline->GetPipelineLayoutHandle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(m_push_constant_data), &m_push_constant_data);

    vkCmdDispatch(command_buffer, ExtendGlobalWorkSize(layer_neuron_count, m_kernel_calc_single_layer_ideal_workgroup_size), batch_count, 1);

    m_dirty_buffers.emplace(layer_output_buffer_vk, BufferSynchronizationEvent::ComputeShaderWrite);
}

std::vector<VkPhysicalDevice> VulkanComputeDevice::GetDeviceList() { return std::vector<VkPhysicalDevice>(); }

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