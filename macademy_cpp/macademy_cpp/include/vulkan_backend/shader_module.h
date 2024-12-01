#pragma once

#include <common.h>

#include <vulkan/vulkan.h>

#include "vulkan_backend/vulkan_instance.h"
#include "vulkan_backend/vulkan_device.h"
#include "SPIRV-Reflect/spirv_reflect.h"

namespace macademy::vk {
class ShaderModule
{
    Device* m_device;
    VkShaderModule m_shader_module;
    std::unordered_map<std::string, uint32_t> m_specialization_constants;

  public:
    ShaderModule(Device* device, const std::string& name, const SpirvBinary& shader_code) : m_device(device)
    {
        ASSERT(shader_code.size() % 4 == 0);

        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = shader_code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(shader_code.data());

        if (vkCreateShaderModule(m_device->GetHandle(), &createInfo, nullptr, &m_shader_module) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create shader module!");
        }

        {
            SpvReflectShaderModule module;
            SpvReflectResult result = spvReflectCreateShaderModule(shader_code.size(), shader_code.data(), &module);
            ASSERT(result == SPV_REFLECT_RESULT_SUCCESS);

            uint32_t spec_constant_count;
            spvReflectEnumerateSpecializationConstants(&module, &spec_constant_count, nullptr);
            std::vector<SpvReflectSpecializationConstant*> spec_constants{spec_constant_count};
            spvReflectEnumerateSpecializationConstants(&module, &spec_constant_count, spec_constants.data());

            for (const auto& spec_constant : spec_constants) {
                m_specialization_constants[spec_constant->name] = spec_constant->constant_id;
            }

            spvReflectDestroyShaderModule(&module);
        }

        device->GetInstance()->SetDebugObjectName(device, uint64_t(m_shader_module), name.c_str(), VK_OBJECT_TYPE_SHADER_MODULE);
    }

    ShaderModule(ShaderModule&&) = delete;

    const std::unordered_map<std::string, uint32_t>& GetSpecializationConstants() const { return m_specialization_constants; }

    VkShaderModule GetHandle() { return m_shader_module; }

    ~ShaderModule() { vkDestroyShaderModule(m_device->GetHandle(), m_shader_module, nullptr); }
};
} // namespace fxgl3::vk

