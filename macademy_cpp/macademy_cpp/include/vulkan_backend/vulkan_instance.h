#pragma once

#include <vulkan/vulkan.h>
#include <vulkan_backend/vulkan_device.h>
#include <vulkan_backend/vulkan_common.h>

#include <sstream>
#include <optional>
#include <algorithm>
#include <cstring>

namespace macademy::vk {
class Device;

class Instance
{
    class DebugMessenger
    {
        PFN_vkCreateDebugUtilsMessengerEXT m_create_func = nullptr;
        PFN_vkDestroyDebugUtilsMessengerEXT m_destroy_func = nullptr;
        VkDebugUtilsMessengerEXT m_debug_messenger;
        VkInstance m_instance;

      public:
        DebugMessenger(VkInstance instance);

        ~DebugMessenger();

        static VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType,
                                                            const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData);
    };

    static bool HasExtension(const std::vector<VkExtensionProperties>& ext_list, std::string_view ext_name);

    static bool HasLayer(const std::vector<VkLayerProperties>& layer_list, std::string_view layer_name);

    static std::vector<VkExtensionProperties> GetInstanceExtensions();

    static std::vector<VkLayerProperties> GetInstanceLayers();

    void InitDevices();

  public:
    Instance(bool enable_validation_layer = false, bool enable_debug_labels = false);

    std::vector<Device*> GetDevices()
    {
        std::vector<Device*> ret;
        ret.reserve(m_devices.size());
        for (const auto& device : m_devices) {
            ret.emplace_back(device.get());
        }

        return ret;
    }

    VkInstance GetHandle() { return m_instance; }

    ~Instance();

    void SetDebugObjectName(Device* device, uint64_t handle, const char* name, VkObjectType type);

    void BeginDebugLabel(VkCommandBuffer command_buffer, const char* name) const;

    void EndDebugLabel(VkCommandBuffer command_buffer) const;

  private:
    VkInstance m_instance;
    std::vector<std::unique_ptr<Device>> m_devices;

    bool m_is_validation_layer_enabled;
    std::unique_ptr<DebugMessenger> m_debug_messenger;
    PFN_vkSetDebugUtilsObjectNameEXT m_set_debug_obj_name_func = nullptr;
    PFN_vkCmdBeginDebugUtilsLabelEXT m_debug_label_begin = nullptr;
    PFN_vkCmdEndDebugUtilsLabelEXT m_debug_label_end = nullptr;
};
} // namespace macademy::vk
