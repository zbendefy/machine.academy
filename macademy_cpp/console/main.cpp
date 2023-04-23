#include <iostream>

#include "network.h"
#include "default_weight_initializer.h"
#include "cpu_compute_backend.h"

int main()
{
    std::vector<macademy::LayerConfig> layers;
    layers.emplace_back(macademy::LayerConfig{.m_activation = macademy::ActivationFunction::Sigmoid, .m_num_neurons = 32});
    layers.emplace_back(macademy::LayerConfig{.m_activation = macademy::ActivationFunction::Sigmoid, .m_num_neurons = 32});
    layers.emplace_back(macademy::LayerConfig{.m_activation = macademy::ActivationFunction::Sigmoid, .m_num_neurons = 4});
    auto network = macademy::NetworkFactory::Build("test", 10, std::span<macademy::LayerConfig>(layers.data(), layers.size()));

    network->GenerateRandomWeights(macademy::DefaultWeightInitializer{});

    macademy::CPUComputeDevice cpu_device{};

    auto uploaded_network = cpu_device.RegisterNetwork(*network);
    std::vector<float> input{1,2,3,4};
    auto result = cpu_device.Evaluate(uploaded_network, input);

    std::cout << "Result:" << std::endl;
    for(auto r : result)
    {
        std::cout << r << std::endl;
    }

    return 0;
}