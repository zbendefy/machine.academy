#include <gtest/gtest.h>

#include "network.h"
#include "default_weight_initializer.h"
#include "cpu_compute_backend.h"
#include "opencl_backend/opencl_compute_device.h"

using namespace macademy;

class ComputeDevicesTest : public ::testing::Test
{
    std::unique_ptr<Network> m_network;

    public:
    ComputeDevicesTest()
    {
        std::vector<LayerConfig> layers;
        layers.emplace_back(LayerConfig{.m_activation = ActivationFunction::Sigmoid, .m_num_neurons = 4});
        layers.emplace_back(LayerConfig{.m_activation = ActivationFunction::ReLU, .m_num_neurons = 15});
        layers.emplace_back(LayerConfig{.m_activation = ActivationFunction::Sigmoid, .m_num_neurons = 2});
        layers.emplace_back(LayerConfig{.m_activation = ActivationFunction::Sigmoid, .m_num_neurons = 2003});
        layers.emplace_back(LayerConfig{.m_activation = ActivationFunction::Sigmoid, .m_num_neurons = 2048});
        layers.emplace_back(LayerConfig{.m_activation = ActivationFunction::Sigmoid, .m_num_neurons = 8});
        m_network = std::make_unique<Network>(NetworkFactory::Build("test", 4, std::span<LayerConfig>(layers.data(), layers.size())));

        m_network->GenerateRandomWeights(DefaultWeightInitializer{});
    }
};

TEST(ComputeDevicesTest, CPUComputeDevice) 
{
    std::vector<float> input{1, 2, 3, 4};

    auto cpu_device = std::make_unique<CPUComputeDevice>();
    auto cpu_device_network = cpu_device->RegisterNetwork(*m_network);

    auto result = cpu_device->Evaluate(*cpu_device_network, input);

    for (auto r : result) {
        std::cout << r << std::endl;
    }
}
