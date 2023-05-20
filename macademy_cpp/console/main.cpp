#include "macademy_utils/console_app.h"

constexpr float PI = 3.141592f;

using namespace std::chrono_literals;
using namespace macademy;

class SineTrainerApp : public ConsoleApp
{
    std::unique_ptr<Network> m_network;
    Training m_trainer;

    //[-pi, pi] --> [0, 1]
    inline float ConvertInputToNetworkInput(float v) const { return (v + PI) / (PI * 2.0f); }

    //[0, 1] --> [-pi, pi]
    inline float ConvertNetworkInputToInput(float v) const { return v * PI * 2 - PI; }

    //[0, 1] --> [-1, 1]
    inline float ConvertNetworkOutputToOutput(float v) const { return v * 2.0f - 1.0f; }

    //[-1, 1] --> [0, 1]
    inline float ConvertOutputToNetworkOutput(float v) const { return v * 0.5f + 0.5f; }

  public:
    SineTrainerApp()
    {
        std::vector<macademy::LayerConfig> layers;
        layers.emplace_back(macademy::LayerConfig{.m_activation = macademy::ActivationFunction::Sigmoid, .m_num_neurons = 128});
        layers.emplace_back(macademy::LayerConfig{.m_activation = macademy::ActivationFunction::Sigmoid, .m_num_neurons = 128});
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

            training_suite->m_mini_batch_size = 100;
            training_suite->m_cost_function = CostFunction::CrossEntropy_Sigmoid;
            training_suite->m_regularization = Regularization::L2;
            training_suite->m_learning_rate = 0.01f;
            training_suite->m_shuffle_training_data = true;
            training_suite->m_epochs = epochs;

            for (int i = 0; i < 10000; ++i) {
                TrainingData training_data;

                const float rnd = (rand() % 1000) / (1000.0f - 1.0f);
                const float sin_input = ConvertNetworkInputToInput(rnd); // random number between [-pi, pi]
                const float sin_output = sinf(sin_input);           // range: [-1, 1]

                training_data.m_input.emplace_back(ConvertInputToNetworkInput(sin_input));
                training_data.m_desired_output.emplace_back(ConvertOutputToNetworkOutput(sin_output));

                training_suite->m_training_data.push_back(training_data);
            }

            auto tracker = m_trainer.Train(*network_on_device->second, *m_selected_device, training_suite);

            std::cout << std::endl;

            TrainingDisplay(*tracker);

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

class SimpleTrainerApp : public ConsoleApp
{
    std::unique_ptr<Network> m_network;
    Training m_trainer;

  public:
    SimpleTrainerApp()
    {
        std::vector<macademy::LayerConfig> layers;
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
            training_suite->m_cost_function = CostFunction::CrossEntropy_Sigmoid;
            training_suite->m_regularization = Regularization::L2;
            training_suite->m_learning_rate = 0.01f;
            training_suite->m_shuffle_training_data = true;
            training_suite->m_epochs = epochs;

            for (int i = 0; i < 1000; ++i) {
                TrainingData training_data;
                const float sin_input = ((rand() % 1000) * 0.001f); // random number between [0, 1]
                const float sin_output = 1.0f - sin_input;          // range: [0, 1]

                training_data.m_input.emplace_back(sin_input);
                training_data.m_desired_output.emplace_back(sin_output);

                training_suite->m_training_data.push_back(training_data);
            }

            auto tracker = m_trainer.Train(*network_on_device->second, *m_selected_device, training_suite);

            TrainingDisplay(*tracker);

            return false;
        };

        m_commands["eval"].m_description = "Eval the paramet on the network";
        m_commands["eval"].m_handler = [this](const std::vector<std::string>& args) {
            float input = 0.0f;
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

            auto result = m_selected_device->Evaluate(*network_on_device->second, std::span<float>(&input, 1));

            std::cout << "Result is: " << result[0];

            return false;
        };
    }
};

int main()
{
    // SineTrainerApp app;
    SimpleTrainerApp app;

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