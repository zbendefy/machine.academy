
#include <network.h>
#include <utils.h>
#include <i_weight_initializer.h>
#include <numeric>

namespace macademy {

const uint32_t Network::BINARY_VERSION = 0x00010000;

std::unique_ptr<Tensor> GenerateWeights(DType dtype, const IWeightInitializer& initializer, uint32_t num_neurons, uint32_t weights_per_neuron)
{
    std::vector<float> data;

    data.reserve(num_neurons * (weights_per_neuron + 1/*bias*/));

    for (size_t i = 0; i < num_neurons; i++)
    {
        for (size_t j = 0; j < weights_per_neuron; j++)
        {
            data.push_back(initializer.GetRandomWeight(weights_per_neuron));
        }
        data.push_back(initializer.GetRandomBias());
    }

    std::array<uint32_t, 1> shape{ uint32_t(data.size()) };

    return std::make_unique<Tensor>(dtype, ToReadOnlyUi8Span(data), shape);
}

std::unique_ptr<Network> BuildSequentialNetwork(const std::string& name, uint32_t input_count, std::span<const LayerConfig> layer_config, const IWeightInitializer& weight_initializer)
{
    if (layer_config.size() < 1 || input_count == 0)
    {
        throw std::runtime_error("BuildSequentialNetwork: invalid layer config sizes");
    }

    std::vector<macademy::Layer> layers;
    uint32_t prev_layer_neuron_count = input_count;
    for (const auto& cfg : layer_config)
    {
        const auto activation_fnc = cfg.m_activation_function;
        layers.emplace_back(macademy::Layer{ .m_tensor = GenerateWeights(DType::Float32, weight_initializer, cfg.m_num_neurons, prev_layer_neuron_count), .m_activation = cfg.m_activation_function, .m_num_neurons = cfg.m_num_neurons});
        prev_layer_neuron_count = cfg.m_num_neurons;
    }

    return std::make_unique<macademy::Network>(name, input_count, layers);
}

Network::Network(const std::string& name, uint32_t input_count, std::span<const Layer> layer_list)
    : m_name(name), m_input_arg_count(input_count)
{
    if (layer_list.empty()) {
        throw std::runtime_error("Error! Cannot create empty network!");
    }

    for (const auto& layer : layer_list)
    {
        auto& back = m_layers.emplace_back();
        back.m_activation = layer.m_activation;
        back.m_num_neurons = layer.m_num_neurons;
        back.m_tensor = std::make_unique<Tensor>(*layer.m_tensor);
    }
}

uint32_t Network::GetNeuronCount() const
{
    return std::accumulate(m_layers.begin(), m_layers.end(), uint32_t(0), [](uint32_t sum, const Layer& layer) { return sum + layer.m_num_neurons; });
}

} // namespace macademy