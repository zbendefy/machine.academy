#include "utils.h"
#include <nlohmann/json.hpp>

namespace macademy {

namespace {
nlohmann::json GetNetworkAsJsonObj(const Network& network)
{
    nlohmann::json output = {};
    output["name"] = network.GetName();
    output["description"] = "";

    uint32_t weights_per_neuron = network.GetInputCount();

    for (const auto& network_layer : network.GetLayers()) {
        nlohmann::json layer;
        nlohmann::json weightsMx;
        nlohmann::json biases;

        auto weights_data = network_layer.m_tensor->AsFloat32().data();
        for (uint32_t n = 0; n < network_layer.m_num_neurons; ++n) {
            nlohmann::json weights;
            for (uint32_t w = 0; w < weights_per_neuron; ++w) {
                weights.push_back(*weights_data);
                ++weights_data;
            }

            biases.push_back(*weights_data);
            ++weights_data;

            weightsMx.push_back(std::move(weights));
        }

        layer["weightMx"] = weightsMx;
        layer["biases"] = biases;

        if (network_layer.m_activation == ActivationFunction::Sigmoid) {
            layer["activationFunction"] = "SigmoidActivation";
        } else if (network_layer.m_activation == ActivationFunction::ReLU) {
            layer["activationFunction"] = "ReLUActivation";
        }

        output["layers"].push_back(layer);

        weights_per_neuron = network_layer.m_num_neurons;
    }

    return output;
}
} // namespace

void ExportNetworkAsJson(const Network& network, std::ostream& stream) { stream << GetNetworkAsJsonObj(network); }

void ExportNetworkAsBson(const Network& network, std::ostream& stream)
{
    auto obj = GetNetworkAsJsonObj(network);
    auto bson_data = nlohmann::json::to_bson(obj);
    stream.write(reinterpret_cast<char*>(bson_data.data()), bson_data.size());
}

void ExportNetworkAsBinary(const Network& network, std::ostream& file)
{

#if 0 //TODOZ
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

    file.write(reinterpret_cast<const char*>(network.GetRawWeightData().data()), network.GetRawWeightData().size() * sizeof(float));
#endif
}
std::unique_ptr<Network> ImportNetworkFromBinary(std::istream& file)
{
#if 0
    uint32_t file_binary_version;
    file.read(reinterpret_cast<char*>(&file_binary_version), sizeof(file_binary_version));

    if (file_binary_version != Network::BINARY_VERSION) {
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
#else
    return {};
#endif
}

} // namespace macademy