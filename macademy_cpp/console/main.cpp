#include <iostream>
#include <chrono>
#include <functional>

#include "network.h"
#include "default_weight_initializer.h"
#include "cpu_compute_backend.h"
#include "opencl_backend/opencl_compute_device.h"
#include "training_suite.h"
#include "training.h"
#include <set>
#include <map>
#include <cmath>

constexpr float PI = 3.141592f;

using namespace macademy;

class ConsoleApp
{
  public:
    struct Command
    {
        std::string m_description;
        std::function<bool(const std::vector<std::string>&)> m_handler;
    };

  protected:
    std::set<std::unique_ptr<IComputeDevice>> m_devices;
    std::map<IComputeDevice*, std::unique_ptr<NetworkResourceHandle>> m_uploaded_networks;
    std::unordered_map<std::string, Command> m_commands;
    IComputeDevice* m_selected_device = nullptr;

    std::vector<std::string> Split(const std::string& src, const char delimiter)
    {
        std::vector<std::string> tokens;
        std::string token;
        std::stringstream ss(src);
        while (getline(ss, token, delimiter)) {
            tokens.push_back(token);
        }

        return tokens;
    }

  public:
    void AddCommand(const std::string& name, Command&& cmd) { m_commands[name] = std::move(cmd); }

    ConsoleApp()
    {
        m_devices.insert(std::make_unique<CPUComputeDevice>());
        m_selected_device = m_devices.begin()->get();

        auto opencl_devices = OpenCLComputeDevice::GetDeviceList();
        for (const auto cl_device : opencl_devices) {
            m_devices.insert(std::make_unique<OpenCLComputeDevice>(cl_device));
        }

        m_commands["quit"].m_description = "Exits the application";
        m_commands["quit"].m_handler = [](const std::vector<std::string>&) { return true; };

        m_commands["help"].m_description = "Displays this help message";
        m_commands["help"].m_handler = [this](const std::vector<std::string>&) {
            for (const auto& it : m_commands) {
                std::cout << it.first << " - " << it.second.m_description << std::endl;
            }
            return false;
        };

        m_commands["list_devices"].m_description = "List info about available compute devices";
        m_commands["list_devices"].m_handler = [this](const std::vector<std::string>&) {
            for (const auto& device : m_devices) {
                if (device.get() == m_selected_device) {
                    std::cout << "* ";
                } else {
                    std::cout << "  ";
                }
                std::cout << device->GetDeviceName() << std::endl;
            }
            return false;
        };

        m_commands["device_info"].m_description = "List info about the currently selected device";
        m_commands["device_info"].m_handler = [this](const std::vector<std::string>&) {
            for (const auto& device : m_devices) {
                std::cout << device->GetDeviceName() << std::endl;
            }
            return false;
        };

        m_commands["device_info"].m_description = "List info about the currently selected device";
        m_commands["device_info"].m_handler = [this](const std::vector<std::string>&) {
            if (m_selected_device) {
                std::cout << "Name: " << m_selected_device->GetDeviceName() << std::endl;
                std::cout << "Compute units: " << m_selected_device->GetComputeUnits() << std::endl;
                std::cout << "Memory: " << (m_selected_device->GetTotalMemory() / (1024 * 1024)) << "MB" << std::endl;
            } else {
                std::cout << "No selected device!";
            }
            return false;
        };
    }

    void Run()
    {
        std::string command_line;

        while (true) {
            command_line.clear();
            std::cout << "> ";
            std::getline(std::cin, command_line);

            auto args = Split(command_line, ' ');

            if (args.empty()) {
                continue;
            }

            auto it = m_commands.find(args[0]);

            if (it != m_commands.end()) {
                if (it->second.m_handler(args)) {
                    break;
                }
            } else {
                std::cout << "No such command: " << args[0];
            }

            std::cout << std::endl;
        }
    }
};

class SineTrainerApp : public ConsoleApp
{
    std::unique_ptr<Network> m_network;
    Training m_trainer;

    //[-pi, pi] --> [0, 1]
    inline float ConvertInputToNetworkInput(float v) const 
    { 
        return (v + PI) / (PI * 2.0f);
    }

    //[0, 1] --> [-1, 1]
    inline float ConvertNetworkOutputToOutput(float v) const 
    {
        return v * 2.0f - 1.0f; 
    }

    //[-1, 1] --> [0, 1]
    inline float ConvertOutputToNetworkOutput(float v) const { return v * 0.5f + 0.5f; }

  public:
    SineTrainerApp() 
    {
        std::vector<macademy::LayerConfig> layers;
        layers.emplace_back(macademy::LayerConfig{.m_activation = macademy::ActivationFunction::Sigmoid, .m_num_neurons = 32});
        layers.emplace_back(macademy::LayerConfig{.m_activation = macademy::ActivationFunction::Sigmoid, .m_num_neurons = 32});
        layers.emplace_back(macademy::LayerConfig{.m_activation = macademy::ActivationFunction::Sigmoid, .m_num_neurons = 1});
        m_network = macademy::NetworkFactory::Build("test", 1, std::span<macademy::LayerConfig>(layers.data(), layers.size()));
        
        m_network->GenerateRandomWeights(macademy::DefaultWeightInitializer{});

        m_commands["train"].m_description = "Train the network";
        m_commands["train"].m_handler = [this](const std::vector<std::string>& args) {
            uint32_t epochs = 1;
            for (int i = 1; i < args.size(); ++i) {
                switch (i) {
                case 1:
                    epochs = atoi(args[i].c_str());
                    break;
                }
            }

            auto network_on_device = m_uploaded_networks.find(m_selected_device);
            if (network_on_device == m_uploaded_networks.end()) {
                m_uploaded_networks[m_selected_device] = m_selected_device->RegisterNetwork(*m_network);
                network_on_device = m_uploaded_networks.find(m_selected_device);
            }

            auto training_suite = std::make_shared<TrainingSuite>();
            
            training_suite->m_mini_batch_size = 50;
            training_suite->m_cost_function = CostFunction::MeanSquared;
            training_suite->m_regularization = Regularization::None;
            training_suite->m_learning_rate = 0.01f;
            training_suite->m_shuffle_training_data = false;
            training_suite->m_epochs = epochs;

            for (int i = 0; i < 10000; ++i) {
                TrainingData training_data;
                const float sin_input = ((rand() % 1000) * 0.002f - 1.0f) * PI;  //random number between [-pi, pi]
                const float sin_output = std::sinf(sin_input); //range: [-1, 1]

                training_data.m_input.emplace_back(ConvertInputToNetworkInput(sin_input));
                training_data.m_desired_output.emplace_back(ConvertOutputToNetworkOutput(sin_output));

                training_suite->m_training_data.push_back(training_data);
            }

            auto tracker = m_trainer.Train(*network_on_device->second, *m_selected_device, training_suite);

            tracker->m_future.wait();

            std::cout << "Training finished!";

            return false;
        };

        m_commands["eval"].m_description = "Eval the paramet on the network";
        m_commands["eval"].m_handler = [this](const std::vector<std::string>& args) {
            float input = 0.5f;
            for (int i = 1; i < args.size(); ++i) {
                switch (i) {
                case 1:
                    input = atof(args[i].c_str());
                    break;
                }
            }

            auto network_on_device = m_uploaded_networks.find(m_selected_device);
            if (network_on_device == m_uploaded_networks.end()) {
                m_uploaded_networks[m_selected_device] = m_selected_device->RegisterNetwork(*m_network);
                network_on_device = m_uploaded_networks.find(m_selected_device);
            }

            input = ConvertInputToNetworkInput(input);

            auto result = m_selected_device->Evaluate(*network_on_device->second, std::span<float>(&input, 1));

            std::cout << "Result is: " << ConvertNetworkOutputToOutput(result[0]);

            return false;
        };
    }



};

int main()
{
    SineTrainerApp app;

    app.AddCommand("test_all_devices", ConsoleApp::Command{.m_description = "Test", .m_handler = [](const std::vector<std::string>&) {
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

                                                                   std::cout << "Result finished in: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms"
                                                                             << std::endl;
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