
#include <network.h>
#include <i_weight_initializer.h>
#include <numeric>

namespace macademy {

const uint32_t Network::BINARY_VERSION = 0x00010000;

Network::Network(const std::string& name, uint32_t input_count, std::span<LayerConfig> layer_config, std::span<float> weights)
    : m_name(name), m_input_arg_count(input_count), m_layers(layer_config.begin(), layer_config.end())
{
    if (layer_config.empty()) {
        throw std::runtime_error("Error! Cannot create empty network!");
    }

    size_t data_size = 0;

    uint32_t layer_input_count = m_input_arg_count;
    for (size_t i = 0; i < m_layers.size(); ++i) {
        const uint32_t current_layer_size = m_layers[i].m_num_neurons;

        const uint32_t num_weights = current_layer_size * layer_input_count;
        const uint32_t num_biases = current_layer_size;

        data_size += num_weights + num_biases;

        layer_input_count = current_layer_size;
    }

    m_data.resize(data_size, 0);

    if (!weights.empty()) {
        if (weights.size() != data_size) {
            throw std::runtime_error("Invalid input weight data size!");
        }

        memcpy(m_data.data(), weights.data(), weights.size_bytes());
    }
}

void Network::GenerateRandomWeights(const IWeightInitializer& weight_initializer)
{
    size_t weight_bias_id = 0;
    for (size_t i = 0; i < m_layers.size(); ++i) // for each layer
    {
        const uint32_t prev_layer_size = i == 0 ? m_input_arg_count : m_layers[i - 1].m_num_neurons;
        const uint32_t current_layer_size = m_layers[i].m_num_neurons;

        for (uint32_t j = 0; j < current_layer_size; ++j) // for each neuron in this layer
        {
            for (uint32_t k = 0; k < prev_layer_size; ++k) // for each neuron in the previous layer
            {
                m_data[weight_bias_id++] = weight_initializer.GetRandomWeight(prev_layer_size);
            }
            m_data[weight_bias_id++] = weight_initializer.GetRandomBias();
        }
    }
}

uint32_t Network::GetNeuronCount() const
{
    return std::accumulate(m_layers.begin(), m_layers.end(), uint32_t(0), [](uint32_t sum, const LayerConfig& layer_conf) { return sum + layer_conf.m_num_neurons; });
}

} // namespace macademy