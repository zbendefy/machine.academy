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
    layers.emplace_back(macademy::LayerConfig{ .m_activation = macademy::ActivationFunction::Sigmoid, .m_num_neurons = 2048 });
    layers.emplace_back(macademy::LayerConfig{ .m_activation = macademy::ActivationFunction::Sigmoid, .m_num_neurons = 2048 });
    layers.emplace_back(macademy::LayerConfig{.m_activation = macademy::ActivationFunction::Sigmoid, .m_num_neurons = 8});
    auto network = macademy::NetworkFactory::Build("test", 4, std::span<macademy::LayerConfig>(layers.data(), layers.size()));

    network->GenerateRandomWeights(macademy::DefaultWeightInitializer{});

    std::vector<std::unique_ptr<macademy::IComputeDevice>> devices;
    std::vector<std::unique_ptr<macademy::NetworkResourceHandle>> uploaded_networks;
    
    {
        auto cpu_device = std::make_unique<macademy::CPUComputeDevice>();
        uploaded_networks.emplace_back(cpu_device->RegisterNetwork(*network));
        devices.emplace_back(std::move(cpu_device));
    }
    
    for (const auto& opencl_device : macademy::OpenCLComputeDevice::GetDeviceList())
    {
        auto device = std::make_unique<macademy::OpenCLComputeDevice>(opencl_device);
        uploaded_networks.emplace_back(device->RegisterNetwork(*network));
        devices.emplace_back(std::move(device));
    }

    std::vector<float> input{ 1,2,3,4 };

    for (size_t i = 0; i < devices.size(); ++i)
    {
        std::cout << devices[i]->GetDeviceName() << std::endl;

        auto start = std::chrono::steady_clock::now();
        auto result = devices[i]->Evaluate(*uploaded_networks[i], input);
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