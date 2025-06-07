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

inline uint32_t CalculateLargestLayerNeuronCount(std::span<const Layer> layer_config)
{
    return std::max_element(layer_config.begin(), layer_config.end(), [](const Layer& a, const Layer& b) { return a.m_num_neurons < b.m_num_neurons; })->m_num_neurons;
}

template <typename T> inline std::span<const uint8_t> AsUint8TSpan(const T& v) { return std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(&v), sizeof(v)); }

template <typename T> std::span<const uint8_t> ToReadOnlyUi8Span(const T& container)
{
    return std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(container.data()), container.size() * sizeof(typename T::value_type));
}
template <typename T> std::span<const uint8_t> ToReadOnlyUi8Span(std::span<const T> container)
{
    return std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(container.data()), container.size_bytes());
}

template <typename T> std::span<uint8_t> ToWriteableUi8Span(T& container)
{
    return std::span<uint8_t>(reinterpret_cast<uint8_t*>(container.data()), container.size() * sizeof(typename T::value_type));
}
template <typename T> std::span<uint8_t> ToWriteableUi8Span(std::span<T> container) { return std::span<uint8_t>(reinterpret_cast<uint8_t*>(container.data()), container.size_bytes()); }


template <typename T> int sign(T val) { return (T(0) < val) - (val < T(0)); }

void ExportNetworkAsJson(const Network& network, std::ostream& stream);

void ExportNetworkAsBson(const Network& network, std::ostream& stream);

void ExportNetworkAsBinary(const Network& network, std::ostream& stream);

std::unique_ptr<Network> ImportNetworkFromBinary(std::istream& file);

} // namespace macademy