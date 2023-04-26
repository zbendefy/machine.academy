#include <iostream>

#include "network.h"
#include "default_weight_initializer.h"
#include "cpu_compute_backend.h"
#include "opencl_backend/opencl_compute_device.h"

int main()
{
    std::vector<macademy::LayerConfig> layers;
    layers.emplace_back(macademy::LayerConfig{ .m_activation = macademy::ActivationFunction::Sigmoid, .m_num_neurons = 4 });
    layers.emplace_back(macademy::LayerConfig{ .m_activation = macademy::ActivationFunction::ReLU, .m_num_neurons = 15 });
    layers.emplace_back(macademy::LayerConfig{ .m_activation = macademy::ActivationFunction::Sigmoid, .m_num_neurons = 2 });
    layers.emplace_back(macademy::LayerConfig{.m_activation = macademy::ActivationFunction::Sigmoid, .m_num_neurons = 128});
    layers.emplace_back(macademy::LayerConfig{.m_activation = macademy::ActivationFunction::Sigmoid, .m_num_neurons = 8});
    auto network = macademy::NetworkFactory::Build("test", 4, std::span<macademy::LayerConfig>(layers.data(), layers.size()));

    network->GenerateRandomWeights(macademy::DefaultWeightInitializer{});

    macademy::CPUComputeDevice compute_device_cpu{};
    macademy::OpenCLComputeDevice compute_device_opencl{macademy::OpenCLComputeDevice::AutoSelectDevice()};

    auto uploaded_network_cpu = compute_device_cpu.RegisterNetwork(*network);
    auto uploaded_network_opencl = compute_device_opencl.RegisterNetwork(*network);

    std::vector<float> input{1,2,3,4};
    auto result_cpu = compute_device_cpu.Evaluate(*uploaded_network_cpu, input);
    auto result_opencl = compute_device_opencl.Evaluate(*uploaded_network_opencl, input);

    std::cout << "Result CPU:" << std::endl;
    for(auto r : result_cpu)
    {
        std::cout << r << std::endl;
    }

    std::cout << std::endl;

    std::cout << "Result OpenCL:" << std::endl;
    for (auto r : result_opencl)
    {
        std::cout << r << std::endl;
    }

    return 0;
}