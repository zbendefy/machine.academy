#pragma once

#include <fstream>
#include <vector>
#include <vulkan/vulkan.h>
#include <vulkan_backend/vulkan_common.h>
#include <vulkan_backend/vulkan_device.h>
#include <vulkan_backend/shader_module.h>

namespace macademy::vk {

class ShaderSpecializationMap;

class ComputePipeline
{
    std::string m_name;
    Device* m_device;
    VkPipelineLayout m_pipeline_layout;
    VkPipeline m_compute_pipeline;

  public:
    ComputePipeline(Device* device, std::string_view name, const ComputePipelineDescriptor& pipeline_desc, const ShaderSpecializationMap& specialization_parameters);

    ~ComputePipeline()
    {
        vkDestroyPipeline(m_device->GetHandle(), m_compute_pipeline, nullptr);
        vkDestroyPipelineLayout(m_device->GetHandle(), m_pipeline_layout, nullptr);
    }

    VkPipeline GetHandle() { return m_compute_pipeline; }

    const std::string& GetName() const { return m_name; }

    VkPipelineLayout GetPipelineLayoutHandle() { return m_pipeline_layout; }
};
} // namespace macademy::vk
