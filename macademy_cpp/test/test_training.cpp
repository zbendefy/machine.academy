#include <gtest/gtest.h>

#include "network.h"
#include "default_weight_initializer.h"
#include "training_suite.h"
#include "cpu_backend/cpu_compute_backend.h"
#ifdef MACADEMY_OPENCL_BACKEND
#include "opencl_backend/opencl_compute_device.h"
#endif
#ifdef MACADEMY_VULKAN_BACKEND
#include "vulkan_backend/vulkan_compute_device.h"
#endif
#include "compute_device_factory.h"
#include "compute_tasks.h"
#include "utils.h"

using namespace macademy;

class TrainingTest : public ::testing::Test
{
  public:
    std::unique_ptr<Network> m_network;
    ComputeTasks m_compute_tasks;

    TrainingTest()
    {
        std::vector<LayerConfig> layers;
        layers.emplace_back(LayerConfig{.m_activation = ActivationFunction::Sigmoid, .m_num_neurons = 32});
        layers.emplace_back(LayerConfig{.m_activation = ActivationFunction::Sigmoid, .m_num_neurons = 32});
        layers.emplace_back(LayerConfig{.m_activation = ActivationFunction::Sigmoid, .m_num_neurons = 32});
        m_network = NetworkFactory::Build("test", 5, std::span<LayerConfig>(layers.data(), layers.size()));

        m_network->GenerateRandomWeights(XavierWeightInitializer{});
    }
};

TEST_F(TrainingTest, Training)
{
    auto cpu_compute_device_info = CPUComputeDevice::GetCpuComputeDeviceInfo();

    auto compute_device = ComputeDeviceFactory::CreateComputeDevice(cpu_compute_device_info);
    auto network_resources = std::make_unique<NetworkResourceHandle>(*m_network, *compute_device);

    /* TrainingSuite ts{};
    m_compute_tasks.Train(*network_resources, ts, 0, 10);*/
}
