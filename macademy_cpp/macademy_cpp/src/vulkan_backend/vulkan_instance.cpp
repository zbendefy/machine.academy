#include <vulkan_backend/vulkan_instance.h>

namespace macademy::vk {

void Instance::BeginDebugLabel(VkCommandBuffer command_buffer, const char* name) const
{
    if (m_debug_label_begin) {
        VkDebugUtilsLabelEXT label_info{};
        label_info.color[0] = 1.0f;
        label_info.color[1] = 1.0f;
        label_info.color[2] = 1.0f;
        label_info.color[3] = 1.0f;
        label_info.pLabelName = name;
        label_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
        (*m_debug_label_begin)(command_buffer, &label_info);
    }
}

void Instance::EndDebugLabel(VkCommandBuffer command_buffer) const
{
    if (m_debug_label_end) {
        (*m_debug_label_end)(command_buffer);
    }
}

bool Instance::HasExtension(const std::vector<VkExtensionProperties>& ext_list, std::string_view ext_name)
{
    bool found = false;

    for (const auto& extensions : ext_list) {
        if (strcmp(ext_name.data(), extensions.extensionName) == 0) {
            found = true;
            break;
        }
    }
    return found;
}

bool Instance::HasLayer(const std::vector<VkLayerProperties>& layer_list, std::string_view layer_name)
{
    bool found = false;

    for (const auto& layer : layer_list) {
        if (strcmp(layer_name.data(), layer.layerName) == 0) {
            found = true;
            break;
        }
    }
    return found;
}

std::vector<VkExtensionProperties> Instance::GetInstanceExtensions()
{
    uint32_t extensionCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
    std::vector<VkExtensionProperties> extensions(extensionCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());

    return extensions;
}

std::vector<VkLayerProperties> Instance::GetInstanceLayers()
{
    uint32_t layerCount = 0;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    return availableLayers;
}

Instance::Instance(bool enable_validation_layer, bool enable_debug_labels) : m_is_validation_layer_enabled(enable_validation_layer)
{
    auto extensions = GetInstanceExtensions();
    auto layers = GetInstanceLayers();

    std::vector<const char*> requested_extensions = {};
    std::vector<const char*> requested_layers = {};

    if (enable_validation_layer) {
        if (!HasLayer(layers, ValidationLayerExtensionStr)) {
            throw std::runtime_error("Validation layer not found!");
        }
        requested_layers.emplace_back(ValidationLayerExtensionStr);
    }

    if (enable_validation_layer || enable_debug_labels) {
        if (HasExtension(extensions, VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
            requested_extensions.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        } else {
            //"Debug util extension not found!"
        }
    }

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
    createInfo.enabledExtensionCount = requested_extensions.size();
    createInfo.ppEnabledExtensionNames = requested_extensions.data();
    createInfo.enabledLayerCount = requested_layers.size();
    createInfo.ppEnabledLayerNames = requested_layers.data();

    if (vkCreateInstance(&createInfo, nullptr, &m_instance) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create vulkan instance!");
    }

    if (enable_validation_layer || enable_debug_labels) {
        m_debug_messenger = std::make_unique<DebugMessenger>(m_instance);
        m_set_debug_obj_name_func = (PFN_vkSetDebugUtilsObjectNameEXT)vkGetInstanceProcAddr(m_instance, "vkSetDebugUtilsObjectNameEXT");
        m_debug_label_begin = (PFN_vkCmdBeginDebugUtilsLabelEXT)vkGetInstanceProcAddr(m_instance, "vkCmdBeginDebugUtilsLabelEXT");
        m_debug_label_end = (PFN_vkCmdEndDebugUtilsLabelEXT)vkGetInstanceProcAddr(m_instance, "vkCmdEndDebugUtilsLabelEXT");
    }
}

Instance::~Instance()
{
    m_debug_messenger.reset();
    vkDestroyInstance(m_instance, nullptr);
}

void Instance::SetDebugObjectName(Device* device, uint64_t handle, const char* name, VkObjectType type)
{
    if ((m_is_validation_layer_enabled || m_set_debug_obj_name_func)) {
        VkDebugUtilsObjectNameInfoEXT obj_name_info{};
        obj_name_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
        obj_name_info.objectHandle = handle;
        obj_name_info.pObjectName = name;
        obj_name_info.objectType = type;
        obj_name_info.pNext = nullptr;

        (*m_set_debug_obj_name_func)(device->GetHandle(), &obj_name_info);
    }
}

Instance::DebugMessenger::DebugMessenger(VkInstance instance) : m_instance(instance)
{
    m_create_func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    m_destroy_func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");

    VkDebugUtilsMessengerCreateInfoEXT createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = DebugCallback;
    createInfo.pUserData = nullptr;

    if (m_create_func != nullptr) {
        m_create_func(m_instance, &createInfo, nullptr, &m_debug_messenger);
    } else {
        throw std::runtime_error("debug util extension not present!");
    }
}

Instance::DebugMessenger::~DebugMessenger()
{
    if (m_destroy_func != nullptr) {
        m_destroy_func(m_instance, m_debug_messenger, nullptr);
    }
}

VKAPI_ATTR VkBool32 VKAPI_CALL Instance::DebugMessenger::DebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType,
                                                                       const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData)
{
    if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        printf("Vulkan validation error: %s\n", pCallbackData->pMessage);
    } else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        printf("Vulkan validation warning: %s\n", pCallbackData->pMessage);
    }
    /*if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        LogError(pCallbackData->pMessage);
    } else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        LogWarning(pCallbackData->pMessage);
    } else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) {
        LogInfo(pCallbackData->pMessage);
    } else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT) {
        LogVerbose(pCallbackData->pMessage);
    }*/

    return VK_FALSE;
}

} // namespace macademy::vk
