#include "utils.h"
#include <nlohmann/json.hpp>

namespace macademy {

void ExportNetworkAsJson(const Network& network, std::ofstream& stream) 
{
    nlohmann::json output = {};
    output["name"] = network.GetName();
    output["description"] = "";

}

void ExportNetworkAsBinary(const Network& network, std::ofstream& file)
{
    file << uint32_t(Network::BINARY_VERSION);

    file << uint32_t(network.GetName().size());
    file << network.GetName();

    file << uint32_t(network.GetLayerConfig().size());
    for (const auto& layer : network.GetLayerConfig()) {
        file << uint32_t(layer.m_activation);
        file << uint32_t(layer.m_num_neurons);
    }

    //write weights and biases
    file.write(reinterpret_cast<const char*>(network.GetRawWeightData().data()), network.GetRawWeightData().size());
}

} // namespace macademy