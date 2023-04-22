#pragma once

#include "common.h"
#include "i_compute_backend.h"

#include <string>

namespace macademy
{
    enum class ActivationFunction
    {
        Passtrough,
        Sigmoid,
        ReLU
    };

    struct LayerConfig
    {
        ActivationFunction m_activation;
        uint32_t m_num_neurons = 0;
    };

    class NetworkFactory;
    class IWeightInitializer;

    class Network
    {
        friend class NetworkFactory;

        std::string m_name;
        std::string m_description;

        std::vector<float> m_data;

        const std::vector<LayerConfig> m_layers;
        const uint32_t m_input_arg_count{};

        public:

        Network(const std::string& name, uint32_t input_count, std::span<LayerConfig> layer_config);

        std::span<const float> GetRawWeightData() const
        {
            return std::span<const float>(m_data.data(), m_data.size());
        }
        
        std::span<float> GetRawWeightData()
        {
            return std::span<float>(m_data.data(), m_data.size());
        }

        std::span<const LayerConfig> GetLayerConfig() const
        {
            return std::span<const LayerConfig>(m_layers.data(), m_layers.size());
        }

        uint32_t GetLayerCount() const
        {
            return uint32_t(m_layers.size());
        }

        void GenerateRandomWeights(const IWeightInitializer& weight_initializer);

        const std::string& GetName() const { return m_name; }
        
        const std::string& GetDescription() const { return m_description; }
    };

    class NetworkFactory
    {
        public:

        std::unique_ptr<Network> Build(const std::string& name, uint32_t input_count, std::span<LayerConfig> layer_config)
        {
            return std::make_unique<Network>(name, input_count, layer_config);
        }

    };

}