#include "macademy_utils/console_app.h"
#include "compute_device_factory.h"
#include "i_compute_device.h"
#include "utils.h"
#include <sstream>

namespace macademy {

enum class ExportMode
{
    Binary,
    Json,
    Bson
};

std::vector<std::string> ConsoleApp::Split(const std::string& src, const char delimiter)
{
    std::vector<std::string> tokens;
    std::string token;
    std::stringstream ss(src);
    while (getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }

    return tokens;
}

void ConsoleApp::TrainingDisplay(const TrainingResultTracker& tracker)
{
    while (true) {
        auto status = tracker.m_future.wait_for(250ms);
        if (status == std::future_status::ready) {
            std::cout << "\rTraining finished!                                          " << std::endl;
            break;
        }

        constexpr int progressbar_length = 20;

        std::cout << "\rCurrent epoch: " << tracker.m_epochs_finished << ", Epoch progress: |";
        for (float i = 0.0f; i < 1.0f; i += (1.0f / progressbar_length)) {
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

void ConsoleApp::EnsureNetworkResources()
{
    if (!m_network_resources) {
        m_network_resources = std::make_unique<NetworkResourceHandle>(*m_network, *m_compute_device);
    }
}

ConsoleApp::ConsoleApp()
{
    m_devices = ComputeDeviceFactory::EnumerateComputeDevices();

    m_selected_device_info = m_devices[0];
    m_compute_device = ComputeDeviceFactory::CreateComputeDevice(m_selected_device_info);

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
            if (device == m_selected_device_info) {
                std::cout << "* " << id << ": ";
            } else {
                std::cout << "  " << id << ": ";
            }
            ++id;
            std::cout << "[" << device.m_backend << "] " << device.m_device_name << std::endl;
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
            m_network_resources.reset();
            m_compute_device.reset();

            m_selected_device_info = m_devices[device_id];
            m_compute_device = ComputeDeviceFactory::CreateComputeDevice(m_selected_device_info);
            std::cout << "Selected device: " << m_selected_device_info.m_device_name << std::endl;
        } else {
            std::cout << "Invalid device id!" << std::endl;
        }

        return false;
    };

    m_commands["device_info"].m_description = "List info about the currently selected device";
    m_commands["device_info"].m_handler = [this](const std::vector<std::string>&) {
        const bool float16_supported = false; // m_selected_device_info.SupportsWeightFormat(NetworkWeightFormat::Float16);
        const bool float32_supported = true;  // m_selected_device_info.SupportsWeightFormat(NetworkWeightFormat::Float32);

        std::cout << "Name: " << m_selected_device_info.m_device_name << std::endl;
        std::cout << "Memory: " << (m_selected_device_info.m_total_memory / (1024 * 1024)) << "MB" << std::endl;
        std::cout << "Supported formats: "
                  << "Float16: " << (float16_supported ? "Yes" : "No") << ", Float32: " << (float32_supported ? "Yes" : "No") << std::endl;

        return false;
    };

    m_commands["benchmark_device"].m_description = "Benchmark the currently selected device";
    m_commands["benchmark_device"].m_handler = [this](const std::vector<std::string>&) {
        if (m_compute_device) {
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

            auto network_resources = std::make_unique<NetworkResourceHandle>(*network, *m_compute_device);
            std::vector<float> input{};
            input.reserve(network->GetInputCount());
            for (int i = 0; i < network->GetInputCount(); ++i) {
                input.emplace_back(float(i) / (network->GetInputCount() - 1));
            }

            std::cout << "Running benchmark on device: " << m_compute_device->GetDeviceName() << std::endl;

            auto start = std::chrono::system_clock::now();
            auto result = m_compute_tasks.Evaluate(*network_resources, input);
            auto end = std::chrono::system_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "Evaluation time: " << elapsed.count() << "ms";

        } else {
            std::cout << "No selected device!";
        }
        return false;
    };

    m_commands["export"].m_description = "Export a neural network to file";
    m_commands["export"].m_handler = [this](const std::vector<std::string>& args) {
        if (!m_network) {
            std::cout << "Error! There is no network to be exported!" << std::endl;
            return false;
        }

        std::string filename = "output.bin";
        ExportMode export_mode = ExportMode::Binary;
        for (int i = 1; i < args.size(); ++i) {
            if (args[i] == "--json") {
                export_mode = ExportMode::Json;
            } else if (args[i] == "--bson") {
                export_mode = ExportMode::Bson;
            } else {
                filename = args[i];
            }
        }

        std::ofstream f{filename, std::ios::out | std::ios::binary};
        if (export_mode == ExportMode::Json) {
            ExportNetworkAsJson(*m_network, f);
        } else if (export_mode == ExportMode::Bson) {
            ExportNetworkAsBson(*m_network, f);
        } else {
            ExportNetworkAsBinary(*m_network, f);
        }
        f.close();
        return false;
    };

    m_commands["import"].m_description = "Import a neural network from file";
    m_commands["import"].m_handler = [this](const std::vector<std::string>& args) {
        std::string filename = "output.bin";
        for (int i = 1; i < args.size(); ++i) {
            filename = args[i];
        }

        std::ifstream f{filename, std::ios::in | std::ios::binary};
        m_network = ImportNetworkFromBinary(f);
        f.close();

        return false;
    };

    m_commands["print_network"].m_description = "Print details about the network";
    m_commands["print_network"].m_handler = [this](const std::vector<std::string>& args) {
        if (m_network) {
            std::cout << m_network->GetName() << std::endl;
            std::cout << "Layers:" << std::endl;

            std::cout << " Input layer: " << m_network->GetInputCount() << std::endl;

            for (int i = 0; i < m_network->GetLayerConfig().size(); ++i) {
                const auto& layer_conf = m_network->GetLayerConfig()[i];
                std::cout << " Layer " << i << ": " << layer_conf.m_num_neurons << "  Activation: " << uint32_t(layer_conf.m_activation) << std::endl;
            }
        } else {
            std::cout << "No loaded network!" << std::endl;
        }

        return false;
    };
}

void ConsoleApp::Run()
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

} // namespace macademy