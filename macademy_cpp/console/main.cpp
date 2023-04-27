#include <iostream>
#include <chrono>

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
    layers.emplace_back(macademy::LayerConfig{ .m_activation = macademy::ActivationFunction::Sigmoid, .m_num_neurons = 16384 });
    layers.emplace_back(macademy::LayerConfig{ .m_activation = macademy::ActivationFunction::Sigmoid, .m_num_neurons = 16384 });
    layers.emplace_back(macademy::LayerConfig{ .m_activation = macademy::ActivationFunction::Sigmoid, .m_num_neurons = 16384 });
    layers.emplace_back(macademy::LayerConfig{ .m_activation = macademy::ActivationFunction::Sigmoid, .m_num_neurons = 16384 });
    layers.emplace_back(macademy::LayerConfig{ .m_activation = macademy::ActivationFunction::Sigmoid, .m_num_neurons = 16384 });
    layers.emplace_back(macademy::LayerConfig{ .m_activation = macademy::ActivationFunction::Sigmoid, .m_num_neurons = 16384 });
    layers.emplace_back(macademy::LayerConfig{ .m_activation = macademy::ActivationFunction::Sigmoid, .m_num_neurons = 16384 });
    layers.emplace_back(macademy::LayerConfig{.m_activation = macademy::ActivationFunction::Sigmoid, .m_num_neurons = 16384});
    layers.emplace_back(macademy::LayerConfig{.m_activation = macademy::ActivationFunction::Sigmoid, .m_num_neurons = 8});
    auto network = macademy::NetworkFactory::Build("test", 4, std::span<macademy::LayerConfig>(layers.data(), layers.size()));

    network->GenerateRandomWeights(macademy::DefaultWeightInitializer{});

    using DeviceNetwork = std::pair<std::unique_ptr<macademy::IComputeDevice>, std::unique_ptr<macademy::NetworkResourceHandle>>;
    std::vector<DeviceNetwork> devices;
    
    {
        auto cpu_device = std::make_unique<macademy::CPUComputeDevice>();
        auto cpu_network = cpu_device->RegisterNetwork(*network);
        devices.emplace_back(DeviceNetwork(std::move(cpu_device), std::move(cpu_network)));
    }
    
    for (const auto& opencl_device : macademy::OpenCLComputeDevice::GetDeviceList())
    {
        auto device = std::make_unique<macademy::OpenCLComputeDevice>(opencl_device);
        auto ocl_network = device->RegisterNetwork(*network);
        devices.emplace_back(DeviceNetwork(std::move(device), std::move(network)));
    }

    std::vector<float> input{ 1,2,3,4 };

    for (auto& device : devices)
    {
        std::cout << device.first->GetDeviceName() << std::endl;

        auto start = std::chrono::steady_clock::now();
        auto result = device.first->Evaluate(*device.second, input);
        auto end = std::chrono::steady_clock::now();

        std::cout << "Result finished in: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        for (auto r : result)
        {
            std::cout << r << std::endl;
        }

        std::cout <<  std::endl;
    }

    return 0;
}