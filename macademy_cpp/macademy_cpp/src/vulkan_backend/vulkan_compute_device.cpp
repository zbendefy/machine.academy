#include "vulkan_backend/vulkan_compute_device.h"
#include "vulkan_backend/vulkan_buffer.h"
#include "network.h"
#include "common.h"
#include "utils.h"
#include "training_suite.h"

#include <fstream>
#include <array>
#include <sstream>

#if !defined(NDEBUG) && RDOC_ENABLED
#define DEBUG_RENDERDOC
#endif

#ifdef DEBUG_RENDERDOC

#include "renderdoc_app.h"

RENDERDOC_API_1_2_0* rdoc_api = NULL;

#endif

namespace macademy {

#define VK_CONSTANTS_HOST
#include "vulkan_backend/shaders/kernel_calc_single_layer_constants.h"
#include "vulkan_backend/shaders/kernel_calc_single_layer.glsl.h"

#include "vulkan_backend/shaders/kernel_training_forward_pass_constants.h"
#include "vulkan_backend/shaders/kernel_training_forward_pass.glsl.h"

#include "vulkan_backend/shaders/kernel_training_backward_pass_constants.h"
#include "vulkan_backend/shaders/kernel_training_backward_pass.glsl.h"
#include "vulkan_backend/shaders/kernel_training_backward_pass_swadd.glsl.h"

#include "vulkan_backend/shaders/kernel_training_apply_gradient_constants.h"
#include "vulkan_backend/shaders/kernel_apply_gradient.glsl.h"

namespace {

size_t GetLocalWorkgroupCount(size_t total_work_items, size_t local_workgroup_size)
{
    return ((total_work_items % local_workgroup_size) == 0) ? (total_work_items / local_workgroup_size) : (total_work_items / local_workgroup_size + 1);
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

KernelResources::KernelResources(vk::Device* device, const std::string& name, uint32_t storage_buffer_count, uint32_t push_constant_size, uint32_t max_descriptor_sets,
                                 const vk::SpirvBinary& spirv_binary, const vk::ShaderSpecializationMap& shader_specialization)
    : m_device(device)
{
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    bindings.resize(storage_buffer_count);

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

    if (vkCreateDescriptorSetLayout(m_device->GetHandle(), &layoutInfo, nullptr, &m_descriptor_set_layout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor set layout!");
    }

    ASSERT(spirv_binary.size() % 4 == 0);

    vk::ComputePipelineDescriptor pipeline_desc{};
    pipeline_desc.m_compute_shader = spirv_binary;
    pipeline_desc.m_compute_shader_entry_function = "main";
    pipeline_desc.m_descriptor_set_layout = m_descriptor_set_layout;
    pipeline_desc.m_push_constant_size = push_constant_size;
    pipeline_desc.m_shader_specialization = shader_specialization;

    m_pipeline = std::make_unique<vk::ComputePipeline>(m_device, name.c_str(), pipeline_desc);

    std::array<VkDescriptorPoolSize, 1> sizes = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, storage_buffer_count * max_descriptor_sets}};

    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.flags = 0;
    pool_info.maxSets = max_descriptor_sets;
    pool_info.poolSizeCount = (uint32_t)sizes.size();
    pool_info.pPoolSizes = sizes.data();

    vkCreateDescriptorPool(m_device->GetHandle(), &pool_info, nullptr, &m_descriptor_pool);
}

KernelResources::~KernelResources()
{
    FreeDescriptorSets();

    if (m_descriptor_set_layout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(m_device->GetHandle(), m_descriptor_set_layout, nullptr);
    }
    if (m_descriptor_pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(m_device->GetHandle(), m_descriptor_pool, nullptr);
    }
}

VkDescriptorSet KernelResources::GetDescriptorSet(const std::vector<const vk::VulkanBuffer*>& storage_buffers)
{
    // descriptor sets bound to specific buffers are cached (as there are many cases where a ping-pong calculation is done that would require the same 2 descriptor sets many times)
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
        vkAllocateDescriptorSets(m_device->GetHandle(), &allocInfo, &descriptor_set);

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

        vkUpdateDescriptorSets(m_device->GetHandle(), descriptor_writes.size(), descriptor_writes.data(), 0, nullptr);

        m_descriptor_sets.emplace(storage_buffers, descriptor_set);

        return descriptor_set;
    } else {
        return it->second;
    }
}

void KernelResources::FreeDescriptorSets()
{
    // Its more efficient to reset the pool, than freeing descriptor sets individually.
    vkResetDescriptorPool(m_device->GetHandle(), m_descriptor_pool, 0);

    m_descriptor_sets.clear();
}

void KernelResources::Bind(VkCommandBuffer command_buffer, const std::vector<const vk::VulkanBuffer*>& buffers, std::span<const uint8_t> push_constant_data)
{
    auto descriptor_set = GetDescriptorSet(buffers);

    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline->GetHandle());
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline->GetPipelineLayoutHandle(), 0, 1, &descriptor_set, 0, 0);

    vkCmdPushConstants(command_buffer, m_pipeline->GetPipelineLayoutHandle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, push_constant_data.size_bytes(), push_constant_data.data());
}

void KernelResources::Dispatch(VkCommandBuffer command_buffer, uint32_t threadgroup_count_x, uint32_t threadgroup_count_y, uint32_t threadgroup_count_z)
{
    vkCmdDispatch(command_buffer, threadgroup_count_x, threadgroup_count_y, threadgroup_count_z);
}

std::vector<ComputeDeviceInfo> VulkanComputeDevice::GetVulkanComputeDeviceInfo()
{
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "macademy";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "foxglove3_macademy";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_1;

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
        cmd_buffer_begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(m_current_command_buffer, &cmd_buffer_begin_info);
    }

    return m_current_command_buffer;
}

VulkanComputeDevice::VulkanComputeDevice(const ComputeDeviceInfo& device_info, const nlohmann::json& device_config)
{
#ifdef DEBUG_RENDERDOC
    if (HMODULE mod = GetModuleHandleA("renderdoc.dll")) {
        pRENDERDOC_GetAPI RENDERDOC_GetAPI = (pRENDERDOC_GetAPI)GetProcAddress(mod, "RENDERDOC_GetAPI");
        int ret = RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_2_0, (void**)&rdoc_api);
        ASSERT(ret == 1);
    }
#endif

#ifdef DEBUG_RENDERDOC
    constexpr bool debug_label_default_enabled = true;
#else
    constexpr bool debug_label_default_enabled = false;
#endif

    bool validation_layer_enabled = GetBoolFlagFromJson(device_config, "validation_layer_enabled", false);
    bool debug_labels_enabled = GetBoolFlagFromJson(device_config, "debug_labels_enabled", debug_label_default_enabled);

    m_kernel_calc_single_layer_ideal_workgroup_size = GetIntFromJson(device_config, "eval_threadgroup_size", m_kernel_calc_single_layer_ideal_workgroup_size);
    m_kernel_training_ideal_workgroup_size_x = GetIntFromJson(device_config, "training_threadgroup_size_x", m_kernel_training_ideal_workgroup_size_x);
    m_kernel_training_ideal_workgroup_size_y = GetIntFromJson(device_config, "training_threadgroup_size_x", m_kernel_training_ideal_workgroup_size_y);
    m_kernel_training_apply_gradient_ideal_workgroup_size = GetIntFromJson(device_config, "gradient_apply_threadgroup_size", m_kernel_training_apply_gradient_ideal_workgroup_size);

    m_instance = std::make_unique<vk::Instance>(validation_layer_enabled, debug_labels_enabled);

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(m_instance->GetHandle(), &deviceCount, nullptr);
    std::vector<VkPhysicalDevice> physical_devices;
    physical_devices.resize(deviceCount);
    vkEnumeratePhysicalDevices(m_instance->GetHandle(), &deviceCount, physical_devices.data());

    m_device = std::make_unique<vk::Device>(m_instance.get(), physical_devices[device_info.m_device_index], true);

    m_hw_atomic_add_support = m_device->GetDeviceAtomicFloatFeatures().shaderBufferFloat32AtomicAdd;

    if (GetBoolFlagFromJson(device_config, "disable_hw_atomic_add", false)) {
        m_hw_atomic_add_support = false;
    }

    {
        vk::ShaderSpecializationMap shader_specialization;
        shader_specialization.emplace(0, m_kernel_calc_single_layer_ideal_workgroup_size);

        // Note: there should be currently at most 2 simultaneous descriptor sets, but I used 8 here just in case the compute tasks api gets used in some unintended way.
        m_kernel_calc_single_layer = std::make_unique<KernelResources>(m_device.get(), "kernel_calc_single_layer", 3, uint32_t(sizeof(CalcSingleLayerPushConstantData)), 8,
                                                                       base64_decode_spirv(vulkan_kernel_source_kernel_calc_single_layer_glsl_b64), shader_specialization);
    }

    {
        vk::ShaderSpecializationMap shader_specialization;
        shader_specialization.emplace(0, m_kernel_training_ideal_workgroup_size_x);
        shader_specialization.emplace(1, m_kernel_training_ideal_workgroup_size_y);

        m_kernel_train_forward_pass = std::make_unique<KernelResources>(m_device.get(), "kernel_train_forward_pass", 4, uint32_t(sizeof(TrainingForwardPassPushConstantData)), 8,
                                                                        base64_decode_spirv(vulkan_kernel_source_kernel_training_forward_pass_glsl_b64), shader_specialization);
    }

    {
        vk::ShaderSpecializationMap shader_specialization;
        shader_specialization.emplace(0, m_kernel_training_ideal_workgroup_size_x);
        shader_specialization.emplace(1, m_kernel_training_ideal_workgroup_size_y);

        const auto spirv_binary =
            base64_decode_spirv(m_hw_atomic_add_support ? vulkan_kernel_source_kernel_training_backward_pass_glsl_b64 : vulkan_kernel_source_kernel_training_backward_pass_swadd_glsl_b64);
        m_kernel_train_backward_pass =
            std::make_unique<KernelResources>(m_device.get(), "kernel_train_backward_pass", 7, uint32_t(sizeof(TrainingBackwardPassPushConstantData)), 8, spirv_binary, shader_specialization);
    }

    {
        vk::ShaderSpecializationMap shader_specialization;
        shader_specialization.emplace(0, m_kernel_training_apply_gradient_ideal_workgroup_size);

        m_kernel_train_apply_gradient = std::make_unique<KernelResources>(m_device.get(), "kernel_train_apply_gradient", 3, uint32_t(sizeof(ApplyGradientPushConstantData)), 8,
                                                                          base64_decode_spirv(vulkan_kernel_source_kernel_apply_gradient_glsl_b64), shader_specialization);
    }
}

VulkanComputeDevice::~VulkanComputeDevice() {}

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
    auto& staging_buffer = m_staging_buffers.emplace_back(m_device->GetLoaderStagingBuffer(src.size()));

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
    auto& staging_buffer = m_staging_buffers.emplace_back(m_device->GetLoaderStagingBuffer(dst.size()));
    auto command_buffer = GetCommandBuffer();

    std::array<const vk::VulkanBuffer*, 1> buffers{{vk_buffer}};
    SynchronizeBuffers(command_buffer, SynchronizationAction::TransferRead, std::span<const vk::VulkanBuffer*>(buffers.begin(), buffers.end()));

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
        vkResetCommandBuffer(m_current_command_buffer, 0);
        m_current_command_buffer = VK_NULL_HANDLE;

        for (auto& it : m_memory_reads) {
            auto src_memory = it.m_host_buffer->m_staging_buffer->MapMemory();
            ASSERT(src_memory); // loader staging buffers should be host_visible, and therefore mappable!
            memcpy(it.m_dst.data(), src_memory, it.m_dst.size_bytes());
            it.m_host_buffer->m_staging_buffer->UnmapMemory();
        }

        m_memory_reads.clear();

        m_kernel_calc_single_layer->FreeDescriptorSets();
        m_kernel_train_forward_pass->FreeDescriptorSets();
        m_kernel_train_backward_pass->FreeDescriptorSets();
        m_kernel_train_apply_gradient->FreeDescriptorSets();

        m_staging_buffers.clear();
        m_dirty_buffers.clear();

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

    const auto CollectBarriers = [this, &buffers](BufferSynchronizationEvent buffer_event, VkAccessFlags src_access_mask, VkAccessFlags dst_access_mask) {
        ASSERT(buffer_event == BufferSynchronizationEvent::TransferWrite && src_access_mask == VK_ACCESS_TRANSFER_WRITE_BIT ||
               buffer_event == BufferSynchronizationEvent::ComputeShaderWrite && src_access_mask == VK_ACCESS_SHADER_WRITE_BIT);

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
    };

    const auto InsertBarrier = [command_buffer](VkDependencyFlags srcStageMask, VkDependencyFlags dstStageMask) {
        if (!buffer_memory_barriers.empty()) {
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
        // A compute shader wants to read a buffer...

        // Add barriers for buffers that are currently being transferred to...
        CollectBarriers(BufferSynchronizationEvent::TransferWrite, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        InsertBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        // Add barriers for buffers that are currently being written by an earlier compute shader...
        CollectBarriers(BufferSynchronizationEvent::ComputeShaderWrite, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        InsertBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    } else if (action == SynchronizationAction::TransferRead) {
        // Host side wants to read back a buffer

        // Add barriers for buffers that are currently being transferred to...
        CollectBarriers(BufferSynchronizationEvent::TransferWrite, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        InsertBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

        // Add barriers for buffers that are currently being written by an earlier compute shader...
        CollectBarriers(BufferSynchronizationEvent::ComputeShaderWrite, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        InsertBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    }

    for (int i = 0; i < int(buffers.size()); ++i) {
        m_dirty_buffers.erase(buffers[i]);
    }
}

void VulkanComputeDevice::QueueEvaluateLayer(const IBuffer* tensor_buffer, const IBuffer* layer_input_buffer, IBuffer* layer_output_buffer, ActivationFunction activation,
    uint32_t layer_input_count, uint32_t layer_neuron_count)
{
    const auto weights_buffer_vk = BufferCast<const vk::VulkanBuffer>(tensor_buffer);
    const auto layer_input_buffer_vk = BufferCast<const vk::VulkanBuffer>(layer_input_buffer);
    auto layer_output_buffer_vk = BufferCast<vk::VulkanBuffer>(layer_output_buffer);

    thread_local std::vector<const vk::VulkanBuffer*> buffers;

    buffers.resize(3);
    buffers[0] = weights_buffer_vk;
    buffers[1] = layer_input_buffer_vk;
    buffers[2] = layer_output_buffer_vk;

    auto command_buffer = GetCommandBuffer();

    SynchronizeBuffers(command_buffer, SynchronizationAction::ComputeShaderRead, std::span<const vk::VulkanBuffer*>(buffers.begin(), buffers.end()));

    CalcSingleLayerPushConstantData push_constant_data;
    push_constant_data.activation = uint32_t(activation);
    push_constant_data.layer_input_count = layer_input_count;
    push_constant_data.layer_neuron_count = layer_neuron_count;

    m_kernel_calc_single_layer->Bind(command_buffer, buffers, AsUint8TSpan(push_constant_data));
    m_kernel_calc_single_layer->Dispatch(command_buffer, GetLocalWorkgroupCount(layer_neuron_count, m_kernel_calc_single_layer_ideal_workgroup_size), 1, 1);

    m_dirty_buffers.emplace(layer_output_buffer_vk, BufferSynchronizationEvent::ComputeShaderWrite);
}

void VulkanComputeDevice::QueueTrainForwardPass(const IBuffer* tensor_buffer, const IBuffer* prev_activations, IBuffer* activations, IBuffer* zvalues, ActivationFunction activation_function,
    uint32_t layer_neuron_count, uint32_t weights_per_neuron, uint32_t num_training_samples)
{

#if 0
    const auto weights_buffer_vk = BufferCast<const vk::VulkanBuffer>(weights_buffer);
    const auto layer_config_buffer_vk = BufferCast<const vk::VulkanBuffer>(layer_config_buffer);
    auto activations_zvalues_buffer_vk = BufferCast<vk::VulkanBuffer>(m_activations_zvalues_buffer);
    const auto input_buffer_vk = BufferCast<const vk::VulkanBuffer>(input_buffer);

    thread_local std::vector<const vk::VulkanBuffer*> buffers;

    buffers.resize(4);
    buffers[0] = weights_buffer_vk;
    buffers[1] = layer_config_buffer_vk;
    buffers[2] = activations_zvalues_buffer_vk;
    buffers[3] = input_buffer_vk;

    auto command_buffer = GetCommandBuffer();

    SynchronizeBuffers(command_buffer, SynchronizationAction::ComputeShaderRead, std::span<const vk::VulkanBuffer*>(buffers.begin(), buffers.end()));

    TrainingForwardPassPushConstantData push_constant_data;
    push_constant_data.current_layer_id = current_layer_id;
    push_constant_data.current_layer_weights_offset = current_layer_weights_offset;
    push_constant_data.numTrainingSamples = num_training_samples;
    push_constant_data.totalActivationCount = total_neuron_count;

    m_kernel_train_forward_pass->Bind(command_buffer, buffers, AsUint8TSpan(push_constant_data));
    m_kernel_train_forward_pass->Dispatch(command_buffer, GetLocalWorkgroupCount(layer_neuron_count, m_kernel_training_ideal_workgroup_size_x),
                                          GetLocalWorkgroupCount(num_training_samples, m_kernel_training_ideal_workgroup_size_y), 1);

    m_dirty_buffers.emplace(activations_zvalues_buffer_vk, BufferSynchronizationEvent::ComputeShaderWrite);
#endif
}

void VulkanComputeDevice::QueueTrainBackwardPass(const IBuffer* next_layer_tensor_buffer, const IBuffer* prev_activations_buffer, const IBuffer* layer_activations_buffer, const IBuffer* layer_zvalues_buffer,
    IBuffer* delta_k_vector_buffer_write, const IBuffer* delta_k_vector_buffer_read, IBuffer* current_layer_gradient_buffer, const IBuffer* desiredOutputsBuffer, uint32_t layer_neuron_count, uint32_t weights_per_neuron, ActivationFunction activation_function, uint32_t numTrainingSamples, CostFunction costFunction, uint32_t next_layer_neuron_count)
{
#if 0
    const auto weights_buffer_vk = BufferCast<const vk::VulkanBuffer>(weights_buffer);
    const auto layer_config_buffer_vk = BufferCast<const vk::VulkanBuffer>(layer_config_buffer);
    const auto activations_zvalues_buffer_vk = BufferCast<const vk::VulkanBuffer>(m_activations_zvalues_buffer);
    const auto input_buffer_vk = BufferCast<const vk::VulkanBuffer>(input_buffer);
    auto delta_k_vector_vk = BufferCast<vk::VulkanBuffer>(delta_k_vector);
    auto gradient_vk = BufferCast<vk::VulkanBuffer>(gradient);
    const auto desiredOutputs_vk = BufferCast<const vk::VulkanBuffer>(desiredOutputs);

    thread_local std::vector<const vk::VulkanBuffer*> buffers;

    buffers.resize(7);
    buffers[0] = weights_buffer_vk;
    buffers[1] = layer_config_buffer_vk;
    buffers[2] = activations_zvalues_buffer_vk;
    buffers[3] = input_buffer_vk;
    buffers[4] = delta_k_vector_vk;
    buffers[5] = gradient_vk;
    buffers[6] = desiredOutputs_vk;

    auto command_buffer = GetCommandBuffer();

    SynchronizeBuffers(command_buffer, SynchronizationAction::ComputeShaderRead, std::span<const vk::VulkanBuffer*>(buffers.begin(), buffers.end()));

    TrainingBackwardPassPushConstantData push_constant_data{};
    push_constant_data.current_layer_id = current_layer_id;
    push_constant_data.layer_count = layer_count;
    push_constant_data.numTrainingSamples = numTrainingSamples;
    push_constant_data.totalActivationCount = total_neuron_count;
    push_constant_data.costFunctionId = uint32_t(costFunction);
    push_constant_data.largest_layer_neuron_count = largest_layer_neuron_count;
    push_constant_data.current_layer_weights_offset = layer_weights_offset;

    m_kernel_train_backward_pass->Bind(command_buffer, buffers, AsUint8TSpan(push_constant_data));
    m_kernel_train_backward_pass->Dispatch(command_buffer, GetLocalWorkgroupCount(layer_neuron_count, m_kernel_training_ideal_workgroup_size_x),
                                           GetLocalWorkgroupCount(numTrainingSamples, m_kernel_training_ideal_workgroup_size_y), 1);

    m_dirty_buffers.emplace(delta_k_vector_vk, BufferSynchronizationEvent::ComputeShaderWrite);
    m_dirty_buffers.emplace(gradient_vk, BufferSynchronizationEvent::ComputeShaderWrite);
#endif
}

void VulkanComputeDevice::QueueApplyGradients(IBuffer* tensor_buffer, const IBuffer* gradient_buffer, uint32_t layer_neuron_count, uint32_t weights_per_neuron, float regularization_term_1, float regularization_term_2, float normalized_learning_rate)
{
#if 0
    auto weights_buffer_vk = BufferCast<vk::VulkanBuffer>(weights_buffer);
    const auto gradient_vk = BufferCast<const vk::VulkanBuffer>(gradient_buffer);
    const auto layer_config_buffer_vk = BufferCast<const vk::VulkanBuffer>(layer_config_buffer);

    thread_local std::vector<const vk::VulkanBuffer*> buffers;

    buffers.resize(3);
    buffers[0] = BufferCast<const vk::VulkanBuffer>(weights_buffer);
    buffers[1] = BufferCast<const vk::VulkanBuffer>(gradient_vk);
    buffers[2] = BufferCast<const vk::VulkanBuffer>(layer_config_buffer_vk);

    auto command_buffer = GetCommandBuffer();

    SynchronizeBuffers(command_buffer, SynchronizationAction::ComputeShaderRead, std::span<const vk::VulkanBuffer*>(buffers.begin(), buffers.end()));

    ApplyGradientPushConstantData push_constant_data{};
    push_constant_data.current_layer_id = current_layer_id;
    push_constant_data.current_layer_weights_offset = current_layer_weights_offset;
    push_constant_data.regularization_term_1 = regularization_term_1;
    push_constant_data.regularization_term_2 = regularization_term_2;
    push_constant_data.normalized_learning_rate = normalized_learning_rate;

    m_kernel_train_apply_gradient->Bind(command_buffer, buffers, AsUint8TSpan(push_constant_data));
    m_kernel_train_apply_gradient->Dispatch(command_buffer, GetLocalWorkgroupCount(layer_neuron_count, m_kernel_calc_single_layer_ideal_workgroup_size), 1, 1);

    m_dirty_buffers.emplace(weights_buffer_vk, BufferSynchronizationEvent::ComputeShaderWrite);
#endif
}

std::string VulkanComputeDevice::GetDeviceName() const { return "Vulkan Device: " + m_device->GetName(); }

size_t VulkanComputeDevice::GetTotalMemory() const { return 0; }

bool VulkanComputeDevice::SupportsWeightFormat(DType format) const
{
    switch (format) {
    case macademy::DType::Float16:
        return m_is_float16_supported;
    case macademy::DType::Float32:
        return true;
    }

    throw std::runtime_error("VulkanComputeDevice::SupportsWeightFormat: Invalid DType  !");
}

} // namespace macademy