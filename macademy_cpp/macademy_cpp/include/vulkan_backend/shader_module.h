#pragma once

#include <common.h>

#include <vulkan/vulkan.h>

#include "vulkan_backend/vulkan_instance.h"
#include "vulkan_backend/vulkan_device.h"

namespace macademy::vk {
class ShaderModule
{
    Device* m_device;
    VkShaderModule m_shader_module;

  public:
    ShaderModule(Device* device, const std::string& name, const SpirvBinary& shader_code) : m_device(device)
    {
        ASSERTM(shader_code.size() % 4 == 0, "Invalid SPIR-V bytecode!");

        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = shader_code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(shader_code.data());

        if (vkCreateShaderModule(m_device->GetHandle(), &createInfo, nullptr, &m_shader_module) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create shader module!");
        }

        device->GetInstance()->SetDebugObjectName(device, uint64_t(m_shader_module), name.c_str(), VK_OBJECT_TYPE_SHADER_MODULE);
    }

    ShaderModule(ShaderModule&&) = delete;

    VkShaderModule GetHandle() { return m_shader_module; }

    ~ShaderModule() { vkDestroyShaderModule(m_device->GetHandle(), m_shader_module, nullptr); }
};
} // namespace macademy::vk
