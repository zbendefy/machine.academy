#pragma once

#include "common.h"
#include "i_compute_backend.h"

#include <string>

namespace macademy
{
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

        //Contains the weights and biases for the network in a compact form
        //The layout is like this:
        //<layer1_neuron1_weight1>, <layer1_neuron1_weight2> ... <layer1_neuron1_weightN>, <layer1_neuron1_bias>, <layer1_neuron2_weight1> ...<layer1_neuron2_weightN>, <layer1_neuron2_bias>, ... <layer2_neuron1_weight1>
        //where there are N neurons in the previous (or input) layers
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

        uint32_t GetWeightByteSize() const
        {
            return sizeof(float);
        }

        void GenerateRandomWeights(const IWeightInitializer& weight_initializer);

        uint32_t GetInputCount() const { return m_input_arg_count; }

        uint32_t GetOutputCount() const { return m_layers[m_layers.size() - 1].m_num_neurons; }

        const std::string& GetName() const { return m_name; }
        
        const std::string& GetDescription() const { return m_description; }
    };

    class NetworkFactory
    {
        public:

        static std::unique_ptr<Network> Build(const std::string& name, uint32_t input_count, std::span<LayerConfig> layer_config)
        {
            return std::make_unique<Network>(name, input_count, layer_config);
        }

    };

}