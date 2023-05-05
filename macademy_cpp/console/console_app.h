#pragma once

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
#include <chrono>
#include <map>
#include <cmath>

using namespace std::chrono_literals;
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
    std::vector<std::unique_ptr<IComputeDevice>> m_devices;
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

    void TrainingDisplay(const TrainingResultTracker& tracker)
    {
        while (true) {
            auto status = tracker.m_future.wait_for(250ms);
            if (status == std::future_status::ready) {
                std::cout << "\rTraining finished!                                          " << std::endl;
                break;
            }

            std::cout << "\rCurrent epoch: " << tracker.m_epochs_finished << ", Epoch progress: |";
            for (float i = 0.0f; i < 1.0f; i += (1.0f / 8.0f)) {
                if ((tracker.m_epoch_progress >= i)) {
                    std::cout << "#";
                } else {
                    std::cout << " ";
                }
            }
            std::cout << "|";
            std::cout.flush();
        }
    }

  public:
    void AddCommand(const std::string& name, Command&& cmd) { m_commands[name] = std::move(cmd); }

    ConsoleApp()
    {
        m_devices.emplace_back(std::make_unique<CPUComputeDevice>());
        m_selected_device = m_devices.begin()->get();

        auto opencl_devices = OpenCLComputeDevice::GetDeviceList();
        for (const auto cl_device : opencl_devices) {
            m_devices.emplace_back(std::make_unique<OpenCLComputeDevice>(cl_device));
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
            uint32_t id = 0;
            for (const auto& device : m_devices) {
                if (device.get() == m_selected_device) {
                    std::cout << "* " << id << ": ";
                } else {
                    std::cout << "  " << id << ": ";
                }
                ++id;
                std::cout << device->GetDeviceName() << std::endl;
            }
            return false;
        };

        m_commands["select_device"].m_description = "Select a compute device";
        m_commands["select_device"].m_handler = [this](const std::vector<std::string>& args) {
            int device_id = -1;
            for (int i = 1; i < args.size(); ++i) {
                switch (i) {
                case 1:
                    device_id = atoi(args[i].c_str());
                    break;
                }
            }

            if (device_id >= 0 && device_id < m_devices.size()) {
                m_selected_device = m_devices[device_id].get();
                std::cout << "Selected device: " << m_selected_device->GetDeviceName() << std::endl;
            } else {
                std::cout << "Invalid device id!" << std::endl;
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

        m_commands["benchmark_device"].m_description = "Benchmark the currently selected device";
        m_commands["benchmark_device"].m_handler = [this](const std::vector<std::string>&) {
            if (m_selected_device) {
                std::cout << "Generating network..." << std::endl;
                std::vector<macademy::LayerConfig> layers;
                layers.emplace_back(macademy::LayerConfig{.m_activation = macademy::ActivationFunction::Sigmoid, .m_num_neurons = 12000});
                layers.emplace_back(macademy::LayerConfig{.m_activation = macademy::ActivationFunction::Sigmoid, .m_num_neurons = 12000});
                layers.emplace_back(macademy::LayerConfig{.m_activation = macademy::ActivationFunction::Sigmoid, .m_num_neurons = 12000});
                layers.emplace_back(macademy::LayerConfig{.m_activation = macademy::ActivationFunction::Sigmoid, .m_num_neurons = 12000});
                layers.emplace_back(macademy::LayerConfig{.m_activation = macademy::ActivationFunction::Sigmoid, .m_num_neurons = 128});
                layers.emplace_back(macademy::LayerConfig{.m_activation = macademy::ActivationFunction::Sigmoid, .m_num_neurons = 8});
                auto network = macademy::NetworkFactory::Build("benchmark", 16, std::span<macademy::LayerConfig>(layers.data(), layers.size()));

                // network->GenerateRandomWeights(macademy::DefaultWeightInitializer{});
                auto uploaded_network = m_selected_device->RegisterNetwork(*network);
                std::vector<float> input{};
                input.reserve(network->GetInputCount());
                for (int i = 0; i < network->GetInputCount(); ++i) {
                    input.emplace_back(float(i) / (network->GetInputCount() - 1));
                }

                std::cout << "Running benchmark on device: " << m_selected_device->GetDeviceName() << std::endl;

                auto start = std::chrono::system_clock::now();
                auto result = m_selected_device->Evaluate(*uploaded_network, input);
                auto end = std::chrono::system_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                std::cout << "Evaluation time: " << elapsed.count() << "ms";

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