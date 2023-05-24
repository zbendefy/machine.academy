#pragma once

#include "network.h"

#include <algorithm>
#include <numeric>
#include <cstdint>
#include <span>
#include <fstream>

namespace macademy {
class LayerConfig;
class Network;

inline uint32_t CalculateLargestLayerNeuronCount(std::span<const LayerConfig> layer_config)
{
    return std::max_element(layer_config.begin(), layer_config.end(), [](const LayerConfig& a, const LayerConfig& b) { return a.m_num_neurons < b.m_num_neurons; })->m_num_neurons;
}

inline uint64_t GetOffsetToLayerWeights(const Network& network, uint32_t layer_id)
{
    uint64_t offset = 0;
    for (size_t i = 0; i < layer_id; ++i) {
        const auto layer_neurons = network.GetLayerConfig()[i].m_num_neurons;
        const auto layer_weights = i == 0 ? network.GetInputCount() : network.GetLayerConfig()[i - 1].m_num_neurons;

        offset += layer_neurons * (layer_weights + 1);
    }

    return offset;
}

inline uint64_t GetOffsetToLayerNeuronCount(std::span<const LayerConfig> layer_config, uint32_t layer_id)
{
    return std::accumulate(layer_config.begin(), layer_config.begin() + layer_id, uint64_t(0), [](uint64_t sum, const LayerConfig& layer_config) { return sum + layer_config.m_num_neurons; });
}

inline uint64_t GetLayerWeightsPerNeuronCount(const Network& network, uint32_t layer_id) { return layer_id == 0 ? network.GetInputCount() : network.GetLayerConfig()[layer_id - 1].m_num_neurons; }

template <typename T> int sign(T val) { return (T(0) < val) - (val < T(0)); }

void ExportNetworkAsJson(const Network& network, std::ostream& stream);

void ExportNetworkAsBinary(const Network& network, std::ostream& stream);

std::unique_ptr<Network> ImportNetworkFromBinary(std::istream& file);

} // namespace macademy