#include <gtest/gtest.h>

#include "network.h"
#include "default_weight_initializer.h"
#include "cpu_compute_backend.h"
#include "opencl_backend/opencl_compute_device.h"
#include "utils.h"

using namespace macademy;

class ComputeDevicesTest : public ::testing::Test
{
  public:
    std::unique_ptr<Network> m_network;

    ComputeDevicesTest()
    {
        std::vector<LayerConfig> layers;
        layers.emplace_back(LayerConfig{.m_activation = ActivationFunction::Sigmoid, .m_num_neurons = 4});
        layers.emplace_back(LayerConfig{.m_activation = ActivationFunction::ReLU, .m_num_neurons = 15});
        layers.emplace_back(LayerConfig{.m_activation = ActivationFunction::Sigmoid, .m_num_neurons = 2});
        layers.emplace_back(LayerConfig{.m_activation = ActivationFunction::Sigmoid, .m_num_neurons = 2003});
        layers.emplace_back(LayerConfig{.m_activation = ActivationFunction::Sigmoid, .m_num_neurons = 2048});
        layers.emplace_back(LayerConfig{.m_activation = ActivationFunction::Sigmoid, .m_num_neurons = 8});
        m_network = NetworkFactory::Build("test", 5, std::span<LayerConfig>(layers.data(), layers.size()));

        m_network->GenerateRandomWeights(DefaultWeightInitializer{});
    }
};

TEST_F(ComputeDevicesTest, Utils) { EXPECT_EQ(2048, CalculateLargestLayerNeuronCount(m_network->GetLayerConfig())); }

TEST_F(ComputeDevicesTest, CPUComputeDevice)
{
    std::vector<float> input{1, -2, 3, -10, 10};

    auto cpu_device = std::make_unique<CPUComputeDevice>();
    auto cpu_device_network = cpu_device->RegisterNetwork(*m_network);

    auto result = cpu_device->Evaluate(*cpu_device_network, input);

    const std::array<float, 8> reference_result = {
        0.71487468, //
        0.58039027, //
        0.39867547, //
        0.54535365, //
        0.37358889, //
        0.72223479, //
        0.31536984, //
        0.70718151, //
    };

    ASSERT_EQ(reference_result.size(), result.size());
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_FLOAT_EQ(reference_result[i], result[i]);
    }
}

TEST_F(ComputeDevicesTest, OpenCLComputeDevice)
{
    std::vector<float> input{1, -2, 3, -10, 10};

    auto cpu_device = std::make_unique<CPUComputeDevice>();
    auto cpu_device_network = cpu_device->RegisterNetwork(*m_network);

    auto reference_result = cpu_device->Evaluate(*cpu_device_network, input);

    auto cl_devices = OpenCLComputeDevice::GetDeviceList();

    for (auto cl_device : cl_devices) {
        auto cl_compute_device = std::make_unique<OpenCLComputeDevice>(cl_device);
        std::cout << "Testing " << cl_compute_device->GetDeviceName() << std::endl;
        auto cl_device_network = cl_compute_device->RegisterNetwork(*m_network);
        auto result = cl_compute_device->Evaluate(*cl_device_network, input);

        ASSERT_EQ(reference_result.size(), result.size());
        for (size_t i = 0; i < result.size(); ++i) {
            EXPECT_FLOAT_EQ(reference_result[i], result[i]);
        }
    }
}