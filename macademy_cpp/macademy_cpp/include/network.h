#pragma once

#include "common.h"

#include <string>

namespace macademy {

class IWeightInitializer;

struct Tensor 
{
    DType m_dtype = DType::Float32;
    std::vector<uint8_t> m_data;
    std::vector<uint32_t> m_shape;

    uint32_t GetElementSize() const { return std::accumulate(m_shape.begin(), m_shape.end(), 1u, [](uint32_t a, uint32_t b) {return a * b; }); };
    size_t GetByteSize() const { return m_data.size(); }
    std::span<const uint8_t> GetRawData() const { return std::span<const uint8_t>(m_data.begin(), m_data.end()); }
    std::span<uint8_t> GetRawData() { return std::span<uint8_t>(m_data.begin(), m_data.end()); }
    std::span<uint32_t> const GetShape() { return std::span<uint32_t>(m_shape.begin(), m_shape.end()); }
    DType GetDType() const { return m_dtype; }
    std::span<float> AsFloat32() { return std::span<float>(reinterpret_cast<float*>(m_data.data()), m_data.size() / sizeof(float)); }
    std::span<const float> AsFloat32() const { return std::span<const float>(reinterpret_cast<const float*>(m_data.data()), m_data.size() / sizeof(float)); }

    Tensor(DType dtype, std::span<const uint8_t> data, std::span<const uint32_t> shape)
        : m_dtype(dtype),
        m_data(data.begin(), data.end()),
        m_shape(shape.begin(), shape.end())
    {
    }

    explicit Tensor(const Tensor& t) : Tensor(t.m_dtype, t.m_data, t.m_shape) {}
};

struct LayerConfig
{
    ActivationFunction m_activation_function;
    uint32_t m_num_neurons = 0;
};

std::unique_ptr<Tensor> GenerateWeights(DType dtype, const IWeightInitializer& initializer, uint32_t num_neurons, uint32_t weights_per_neuron);

struct Layer
{
    std::unique_ptr<Tensor> m_tensor;
    ActivationFunction m_activation;
    uint32_t m_num_neurons = 0;
};

class Network
{
    std::string m_name;
    std::string m_description;

    std::vector<Layer> m_layers;
    const uint32_t m_input_arg_count{};

  public:
    Network(const std::string& name, uint32_t input_count, std::span<const Layer> layers);

    std::span<const Layer> GetLayers() const { return std::span<const Layer>(m_layers.data(), m_layers.size()); }

    std::span<Layer> GetLayers() { return std::span<Layer>(m_layers.data(), m_layers.size()); }

    uint32_t GetLayerCount() const { return uint32_t(m_layers.size()); }

    uint32_t GetInputCount() const { return m_input_arg_count; }

    uint32_t GetOutputCount() const { return m_layers[m_layers.size() - 1].m_num_neurons; }

    uint32_t GetNeuronCount() const;

    const std::string& GetName() const { return m_name; }

    const std::string& GetDescription() const { return m_description; }

    static const uint32_t BINARY_VERSION;
};

std::unique_ptr<Network> BuildSequentialNetwork(const std::string& name, uint32_t input_count, std::span<const LayerConfig> layer_config, const IWeightInitializer& weight_initializer);

} // namespace macademy