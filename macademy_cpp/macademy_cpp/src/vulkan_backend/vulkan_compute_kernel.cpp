#include <vulkan_backend/vulkan_compute_kernel.h>
#include <vulkan_backend/vulkan_device.h>
#include <vulkan_backend/vulkan_instance.h>

namespace macademy::vk {

ComputeKernel::ComputeKernel(vk::Device* device, const std::string& name, uint32_t storage_buffer_count, uint32_t push_constant_size, uint32_t max_descriptor_sets, const vk::SpirvBinary& spirv_binary,
                             const vk::ShaderSpecializationMap& shader_specialization)
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

ComputeKernel::~ComputeKernel()
{
    FreeDescriptorSets();

    if (m_descriptor_set_layout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(m_device->GetHandle(), m_descriptor_set_layout, nullptr);
    }
    if (m_descriptor_pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(m_device->GetHandle(), m_descriptor_pool, nullptr);
    }
}

VkDescriptorSet ComputeKernel::GetDescriptorSet(const std::vector<const vk::VulkanBuffer*>& storage_buffers)
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

void ComputeKernel::FreeDescriptorSets()
{
    // Its more efficient to reset the pool, than freeing descriptor sets individually.
    vkResetDescriptorPool(m_device->GetHandle(), m_descriptor_pool, 0);

    m_descriptor_sets.clear();
}

void ComputeKernel::Bind(VkCommandBuffer command_buffer, const std::vector<const vk::VulkanBuffer*>& buffers, std::span<const uint8_t> push_constant_data)
{
    auto descriptor_set = GetDescriptorSet(buffers);

    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline->GetHandle());
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline->GetPipelineLayoutHandle(), 0, 1, &descriptor_set, 0, 0);

    vkCmdPushConstants(command_buffer, m_pipeline->GetPipelineLayoutHandle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, push_constant_data.size_bytes(), push_constant_data.data());
}

void ComputeKernel::Dispatch(VkCommandBuffer command_buffer, uint32_t threadgroup_count_x, uint32_t threadgroup_count_y, uint32_t threadgroup_count_z)
{
    vkCmdDispatch(command_buffer, threadgroup_count_x, threadgroup_count_y, threadgroup_count_z);
}

} // namespace macademy::vk
