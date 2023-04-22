#include <network.h>
#include <i_weight_initializer.h>

namespace macademy
{
    Network::Network(const std::string& name, uint32_t input_count, std::span<LayerConfig> layer_config)
        : m_name(name)
        , m_input_arg_count(input_count)
        , m_layers(layer_config.begin(), layer_config.end())
    {
        size_t data_size = 0;

        for(size_t i = 0; i < m_layers.size(); ++i)
        {
            const uint32_t current_layer_size = m_layers[i].m_num_neurons;
            data_size += current_layer_size + 1;
        }

        m_data.resize(data_size, 0);
    }
    
    void Network::GenerateRandomWeights(const IWeightInitializer& weight_initializer)
    {
        size_t weight_bias_id = 0;
        for(size_t i = 0; i < m_layers.size(); ++i)
        {
            const uint32_t prev_layer_size = i == 0 ? m_input_arg_count : m_layers[i - 1].m_num_neurons;
            const uint32_t current_layer_size = m_layers[i].m_num_neurons;

            for(uint32_t j = 0; j < current_layer_size; ++j)
            {
                m_data[weight_bias_id++] = weight_initializer.GetRandomWeight(prev_layer_size);
            }
            m_data[weight_bias_id++] = weight_initializer.GetRandomBias();
        }
    }

}