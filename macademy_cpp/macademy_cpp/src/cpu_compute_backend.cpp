#include "cpu_compute_backend.h"
#include "network.h"
#include "common.h"

namespace macademy
{
    NetworkResourceHandle CPUComputeDevice::RegisterNetwork(Network& network)
    {
        return NetworkResourceHandle{&network};
    }

    std::vector<float> CPUComputeDevice::Evaluate(const NetworkResourceHandle& network_handle, const std::vector<float>& input) const
    {
        Network& network = *network_handle.m_network;
        auto layer_config = network.GetLayerConfig();
        std::vector<float> layer_args = input;
        std::vector<float> layer_result{};
        float* neuron_weight_data = network.GetRawWeightData().data();
        for(size_t i = 0; i < layer_config.size(); ++i)
        {
            layer_result.clear();
            const uint32_t input_num = uint32_t(layer_args.size());
            const uint32_t output_num = layer_config[i].m_num_neurons;

            layer_result.resize(output_num);
            for(uint32_t j = 0; j < output_num; ++j)
            {
                float acc = 0.0f;
                for (uint32_t k = 0; k < input_num; k++)
                {
                    acc += *neuron_weight_data * layer_args[k];
                    ++neuron_weight_data;
                }
                acc += *neuron_weight_data;
                ++neuron_weight_data;
            }
            std::swap(layer_args, layer_result);
        }

        ASSERTM(neuron_weight_data - network.GetRawWeightData().data() == network.GetRawWeightData().size(), "");
        return layer_args;
    }
}