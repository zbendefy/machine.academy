#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <string>

namespace macademy::vk {
    
using SpirvBinary = std::vector<char>;

constexpr const char* ValidationLayerExtensionStr = "VK_LAYER_KHRONOS_validation";

struct ComputePipelineDescriptor
{
    SpirvBinary m_compute_shader;
    std::string m_compute_shader_entry_function = "main";

    VkDescriptorSetLayout m_descriptor_set_layout{};

    uint32_t m_push_constant_offset = 0;
    uint32_t m_push_constant_size = 0;
};
}