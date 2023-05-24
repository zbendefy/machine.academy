#include "utils.h"
#include <nlohmann/json.hpp>

namespace macademy {

void ExportNetworkAsJson(const Network& network, std::ostream& stream)
{
    nlohmann::json output = {};
    output["name"] = network.GetName();
    output["description"] = "";
}

void ExportNetworkAsBinary(const Network& network, std::ostream& file)
{
    file.write(reinterpret_cast<const char*>(&Network::BINARY_VERSION), sizeof(Network::BINARY_VERSION));

    uint32_t network_name_length = uint32_t(network.GetName().size());
    file.write(reinterpret_cast<const char*>(&network_name_length), sizeof(network_name_length));

    file << network.GetName();

    uint32_t input_count = network.GetInputCount();
    file.write(reinterpret_cast<const char*>(&input_count), sizeof(input_count));

    uint32_t layer_count = uint32_t(network.GetLayerConfig().size());
    file.write(reinterpret_cast<const char*>(&layer_count), sizeof(layer_count));
    for (const auto& layer : network.GetLayerConfig()) {

        uint32_t activation = uint32_t(layer.m_activation);
        file.write(reinterpret_cast<const char*>(&activation), sizeof(activation));

        uint32_t neuron_count = uint32_t(layer.m_num_neurons);
        file.write(reinterpret_cast<const char*>(&neuron_count), sizeof(neuron_count));
    }

    // write weights and biases
    uint64_t total_weight_count = uint64_t(network.GetRawWeightData().size());
    file.write(reinterpret_cast<const char*>(&total_weight_count), sizeof(total_weight_count));

    file.write(reinterpret_cast<const char*>(network.GetRawWeightData().data()), network.GetRawWeightData().size());
}

std::unique_ptr<Network> ImportNetworkFromBinary(std::istream& file)
{
    uint32_t file_binary_version;
    file.read(reinterpret_cast<char*>(&file_binary_version), sizeof(file_binary_version));

    if(file_binary_version != Network::BINARY_VERSION)
    {
        return nullptr;
    }

    uint32_t name_length;
    file.read(reinterpret_cast<char*>(&name_length), sizeof(name_length));

    std::string name;
    name.resize(name_length);
    file.read(name.data(), name_length);

    uint32_t input_count;
    file.read(reinterpret_cast<char*>(&input_count), sizeof(input_count));

    uint32_t layer_count;
    file.read(reinterpret_cast<char*>(&layer_count), sizeof(layer_count));

    std::vector<LayerConfig> layer_config;
    layer_config.reserve(layer_count);
    for (uint32_t i = 0; i < layer_count; ++i) {
        uint32_t activation;
        file.read(reinterpret_cast<char*>(&activation), sizeof(activation));

        uint32_t neuron_count;
        file.read(reinterpret_cast<char*>(&neuron_count), sizeof(neuron_count));

        layer_config.emplace_back(LayerConfig{.m_activation = ActivationFunction(activation), .m_num_neurons = neuron_count});
    }

    uint64_t total_weight_count;
    file.read(reinterpret_cast<char*>(&total_weight_count), sizeof(total_weight_count));
    
    std::vector<float> weights;
    weights.resize(total_weight_count);
    file.read(reinterpret_cast<char*>(weights.data()), total_weight_count * sizeof(float));

    return NetworkFactory::Build(name, input_count, layer_config, weights);
}

} // namespace macademy