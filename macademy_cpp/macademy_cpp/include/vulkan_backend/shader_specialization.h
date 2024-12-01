#pragma once

#include <vulkan/vulkan.h>

#include "vulkan_backend/vulkan_instance.h"
#include "vulkan_backend/vulkan_device.h"

namespace macademy::vk {
class ShaderSpecializationMap
{
    union Data
    {
        uint32_t m_uint;
        int32_t m_int;
        float m_float;
    };

    static_assert(sizeof(Data) == 4);

  public:
    const void* GetDataBase() const { return m_data.data(); }

    std::optional<uint32_t> GetDataOffset(const std::string& name) const
    {
        auto it = m_map.find(name);
        if (it != m_map.end()) {
            return it->second * uint32_t(sizeof(Data));
        }
        return {};
    }

    size_t GetDataSizeBytes() const { return m_data.size() * sizeof(Data); }

    void Setter(Data& data, float value) { data.m_float = value; }

    void Setter(Data& data, uint32_t value) { data.m_uint = value; }

    void Setter(Data& data, int32_t value) { data.m_int = value; }

    void Getter(const Data& data, int32_t& target) { target = data.m_int; }

    void Getter(const Data& data, uint32_t& target) { target = data.m_uint; }

    void Getter(const Data& data, float& target) { target = data.m_float; }

    template <typename T> std::optional<T> GetValue(const std::string& name)
    {
        auto it = m_map.find(name);
        if (it != m_map.end()) {
            T ret;
            Getter(m_data[it->second], ret);
            return ret;
        }
        return {};
    }

    template <typename T> void SetValue(const std::string& name, T value)
    {
        auto it = m_map.emplace(name, 0);
        if (it.second) // there was an insert
        {
            it.first->second = uint32_t(m_data.size());
            Data d;
            Setter(d, value);
            m_data.emplace_back(d);
        } else {
            Setter(m_data[it.first->second], value);
        }
    }

  private:
    std::unordered_map<std::string, uint32_t> m_map;
    std::vector<Data> m_data;
};
} // namespace fxgl3::vk