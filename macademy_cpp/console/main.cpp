#include <iostream>
#include <chrono>
#include <functional>

#include "network.h"
#include "default_weight_initializer.h"
#include "cpu_compute_backend.h"
#include "opencl_backend/opencl_compute_device.h"

using namespace macademy;

class ConsoleApp
{
public:
    struct Command
    {
        std::string m_description;
        std::function<bool(const std::string&)> m_handler;
    };
private:

    std::unordered_map<std::unique_ptr<IComputeDevice>, std::unique_ptr<NetworkResourceHandle>> m_devices;
    std::unordered_map<std::string, Command> m_commands;
    IComputeDevice* m_selected_device = nullptr;

    public:

    void AddCommand(const std::string& name, Command&& cmd)
    {
        m_commands[name] = std::move(cmd);
    }
    
    ConsoleApp()
    {
        m_devices[std::make_unique<CPUComputeDevice>()] = nullptr;
        m_selected_device = m_devices.begin()->first.get();
        
        auto opencl_devices = OpenCLComputeDevice::GetDeviceList();
        for(const auto cl_device : opencl_devices)
        {
            m_devices[std::make_unique<OpenCLComputeDevice>(cl_device)] = nullptr;
        }

        m_commands["quit"].m_description = "Exits the application";
        m_commands["quit"].m_handler = [](const std::string&){return true;};

        m_commands["help"].m_description = "Displays this help message";
        m_commands["help"].m_handler = [this](const std::string&){
            for(const auto& it : m_commands)
            {
                std::cout << it.first << " - " << it.second.m_description << std::endl;
            }
            return false;
        };
        
        m_commands["list_devices"].m_description = "List info about available compute devices";
        m_commands["list_devices"].m_handler = [this](const std::string&){
            for(const auto& device : m_devices)
            {
                if(device.first.get() == m_selected_device)
                {
                    std::cout << "* ";
                }
                else
                {
                    std::cout << "  ";
                }
                std::cout << device.first->GetDeviceName() << std::endl;
            }
            return false;
        };
        
        m_commands["device_info"].m_description = "List info about the currently selected device";
        m_commands["device_info"].m_handler = [this](const std::string&){
            for(const auto& device : m_devices)
            {
                std::cout << device.first->GetDeviceName() << std::endl;
            }
            return false;
        };
        
        m_commands["device_info"].m_description = "List info about the currently selected device";
        m_commands["device_info"].m_handler = [this](const std::string&){
            if(m_selected_device)
            {
                std::cout << "Name: " << m_selected_device->GetDeviceName() << std::endl;
                std::cout << "Compute units: " << m_selected_device->GetComputeUnits() << std::endl;
                std::cout << "Memory: " << (m_selected_device->GetTotalMemory() / (1024*1024)) << "MB" << std::endl;
            }
            else
            {
                std::cout << "No selected device!";
            }
            return false;
        };
    }

    void Run()
    {
        std::string command_line;

        while(true)
        {
            command_line.clear();
            std::cout << "> ";
            std::cin >> command_line;

            auto first_command = command_line.find(' ');

            std::string command_name = (first_command == std::string::npos) ? command_line : (command_line.substr(0, first_command));

            auto it = m_commands.find(command_name);

            if(it != m_commands.end())
            {
                if(it->second.m_handler(command_line))
                {
                    break;
                }
            }
            else
            {
                std::cout << "No such command: " << command_name;
            }

            command_line.clear();
            std::cout << std::endl;
        }
    }
};

int main()
{
    ConsoleApp app;
    app.AddCommand("test_all_devices", ConsoleApp::Command{.m_description = "Test", .m_handler = [](const std::string& command_line){
        std::vector<macademy::LayerConfig> layers;
        layers.emplace_back(macademy::LayerConfig{.m_activation = macademy::ActivationFunction::Sigmoid, .m_num_neurons = 4});
        layers.emplace_back(macademy::LayerConfig{.m_activation = macademy::ActivationFunction::ReLU, .m_num_neurons = 15});
        layers.emplace_back(macademy::LayerConfig{.m_activation = macademy::ActivationFunction::Sigmoid, .m_num_neurons = 2});
        layers.emplace_back(macademy::LayerConfig{.m_activation = macademy::ActivationFunction::Sigmoid, .m_num_neurons = 2048});
        layers.emplace_back(macademy::LayerConfig{.m_activation = macademy::ActivationFunction::Sigmoid, .m_num_neurons = 2048});
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

        for (const auto& opencl_device : macademy::OpenCLComputeDevice::GetDeviceList()) {
            auto device = std::make_unique<macademy::OpenCLComputeDevice>(opencl_device);
            uploaded_networks.emplace_back(device->RegisterNetwork(*network));
            devices.emplace_back(std::move(device));
        }

        std::vector<float> input{1, 2, 3, 4};

        for (size_t i = 0; i < devices.size(); ++i) {
            std::cout << devices[i]->GetDeviceName() << std::endl;
            std::cout << "  Compute units: " << devices[i]->GetComputeUnits() << std::endl;
            std::cout << "  Total memory: " << (devices[i]->GetTotalMemory() / (1024 * 1024)) << "MB" << std::endl;

            auto start = std::chrono::steady_clock::now();
            auto result = devices[i]->Evaluate(*uploaded_networks[i], input);
            auto end = std::chrono::steady_clock::now();

            std::cout << "Result finished in: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
            for (auto r : result) {
                std::cout << r << std::endl;
            }

            std::cout << std::endl;
        }
        return false;
    }});

    app.Run();
    return 0;
}