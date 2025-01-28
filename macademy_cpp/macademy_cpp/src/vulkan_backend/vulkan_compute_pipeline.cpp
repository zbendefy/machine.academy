
#include <vulkan_backend/vulkan_compute_pipeline.h>
#include <vulkan_backend/shader_module.h>
#include <array>

namespace macademy::vk {

struct ShaderInfo
{
    ShaderInfo(Device* device, const std::string& name, const SpirvBinary& shader, VkShaderStageFlagBits stage_flag, const std::string& entry_function,
               const ShaderSpecializationMap& specialization_parameters)
        : m_shader_module(device, name, shader), m_entry_function(entry_function)
    {
        constexpr uint32_t data_size = sizeof(uint32_t);

        m_spec_constant_data.reserve(specialization_parameters.size());

        for (const auto& spec_const : specialization_parameters) {
            m_specialization_entries.emplace_back(VkSpecializationMapEntry{.constantID = spec_const.first, .offset = uint32_t(m_spec_constant_data.size()) * data_size, .size = data_size});
            m_spec_constant_data.emplace_back(spec_const.second);
        }

        m_spec_info.pMapEntries = m_specialization_entries.data();
        m_spec_info.mapEntryCount = m_specialization_entries.size();
        m_spec_info.pData = m_spec_constant_data.data();
        m_spec_info.dataSize = m_spec_constant_data.size() * sizeof(uint32_t);

        m_shader_stage_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        m_shader_stage_create_info.stage = stage_flag;
        m_shader_stage_create_info.module = m_shader_module.GetHandle();
        m_shader_stage_create_info.pName = m_entry_function.c_str();
        m_shader_stage_create_info.pSpecializationInfo = &m_spec_info;
    }

    ShaderModule m_shader_module;
    VkSpecializationInfo m_spec_info{};
    std::vector<VkSpecializationMapEntry> m_specialization_entries;
    std::vector<uint32_t> m_spec_constant_data;
    const std::string& m_entry_function;
    VkPipelineShaderStageCreateInfo m_shader_stage_create_info{};
};

ComputePipeline::ComputePipeline(Device* device, const std::string& name, const ComputePipelineDescriptor& pipeline_desc) : m_name(name), m_device(device)
{
    ShaderInfo cs_info{device, name, pipeline_desc.m_compute_shader, VK_SHADER_STAGE_COMPUTE_BIT, pipeline_desc.m_compute_shader_entry_function, pipeline_desc.m_shader_specialization};

    VkPushConstantRange push_constant_range{};
    push_constant_range.offset = pipeline_desc.m_push_constant_offset;
    push_constant_range.size = pipeline_desc.m_push_constant_size;
    push_constant_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &pipeline_desc.m_descriptor_set_layout;
    pipelineLayoutInfo.pushConstantRangeCount = push_constant_range.size > 0 ? 1 : 0;
    pipelineLayoutInfo.pPushConstantRanges = &push_constant_range;

    if (vkCreatePipelineLayout(m_device->GetHandle(), &pipelineLayoutInfo, nullptr, &m_pipeline_layout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create pipeline layout!");
    }

    device->GetInstance()->SetDebugObjectName(device, uint64_t(m_pipeline_layout), m_name.c_str(), VK_OBJECT_TYPE_PIPELINE_LAYOUT);

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = cs_info.m_shader_stage_create_info;
    pipelineInfo.layout = m_pipeline_layout;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // Optional
    pipelineInfo.basePipelineIndex = -1;              // Optional

    auto result = vkCreateComputePipelines(m_device->GetHandle(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_compute_pipeline);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("failed to create compute pipeline! " + std::to_string(result));
    }

    device->GetInstance()->SetDebugObjectName(device, uint64_t(m_compute_pipeline), m_name.c_str(), VK_OBJECT_TYPE_PIPELINE);
}

} // namespace macademy::vk
