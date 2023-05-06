#include "macademy_utils/console_app.h"

#include <fstream>

using namespace std::chrono_literals;
using namespace macademy;

class MnistTrainerApp : public ConsoleApp
{
    std::unique_ptr<Network> m_network;
    Training m_trainer;
    std::shared_ptr<TrainingSuite> m_training_suite;
    std::vector<TrainingData> m_test_data;

    static std::vector<uint8_t> ReadFile(const std::string& filename)
    {
        std::ifstream infile(filename);
        infile.seekg(0, std::ios::end);
        size_t length = infile.tellg();
        infile.seekg(0, std::ios::beg);

        std::vector<uint8_t> ret;
        ret.resize(length);

        infile.read(reinterpret_cast<char*>(ret.data()), length);

        return ret;
    }

    static void LoadTrainingData(std::vector<TrainingData>& training_data_vector, const std::string& img_filename, const std::string& label_filename)
    {
        // Details: http://yann.lecun.com/exdb/mnist/

        const auto img_data = ReadFile(img_filename);
        const auto label_data = ReadFile(label_filename);

        const size_t data_count = label_data.size() - 8;

        const uint8_t* labels = label_data.data() + 8; // first 8 bytes contain metadata
        const uint8_t* pixels = img_data.data() + 16;  // first 8 bytes contain metadata

        const size_t image_data_size = 28 * 28 * sizeof(uint8_t);

        training_data_vector.reserve(training_data_vector.size() + data_count);

        for (size_t i = 0; i < data_count; ++i) {
            auto& training_data = training_data_vector.emplace_back();

            training_data.m_desired_output = std::vector<float>(10, 0.0f);
            training_data.m_desired_output[labels[i]] = 1.0f;

            training_data.m_input.resize(28 * 28, 0.0f);
            for (uint32_t px = 0; px < 28 * 28; ++px) {
                training_data.m_input[px] = float(pixels[i * image_data_size + px]) / 255.0f; // 0 means background (white), 255 means foreground (black).
            }
        }
    }

  public:
    MnistTrainerApp(const std::string& data_folder)
    {
        std::vector<macademy::LayerConfig> layers;
        layers.emplace_back(macademy::LayerConfig{.m_activation = macademy::ActivationFunction::Sigmoid, .m_num_neurons = 64});
        layers.emplace_back(macademy::LayerConfig{.m_activation = macademy::ActivationFunction::Sigmoid, .m_num_neurons = 10});
        m_network = macademy::NetworkFactory::Build("MNIST digit recognizer", 28 * 28, std::span<macademy::LayerConfig>(layers.data(), layers.size()));

        m_network->GenerateRandomWeights(macademy::DefaultWeightInitializer{});

        m_training_suite = std::make_shared<TrainingSuite>();
        m_training_suite->m_mini_batch_size = 100;
        m_training_suite->m_cost_function = CostFunction::CrossEntropy_Sigmoid;
        m_training_suite->m_regularization = Regularization::L2;
        m_training_suite->m_learning_rate = 0.01f;
        m_training_suite->m_shuffle_training_data = true;

        LoadTrainingData(m_training_suite->m_training_data, data_folder + "/train-images.idx3-ubyte", data_folder + "/train-labels.idx1-ubyte");

        LoadTrainingData(m_test_data, data_folder + "/t10k-images.idx3-ubyte", data_folder + "/t10k-labels.idx1-ubyte");

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

            m_training_suite->m_epochs = epochs;

            auto tracker = m_trainer.Train(*network_on_device->second, *m_selected_device, m_training_suite);

            std::cout << std::endl;

            TrainingDisplay(*tracker);

            return false;
        };

        m_commands["eval"].m_description = "Eval the test input [n] on the network";
        m_commands["eval"].m_handler = [this](const std::vector<std::string>& args) {
            int input = 0;
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

            if (input < m_test_data.size() && input > 0) {
                auto& test_data = m_test_data[input];
                for (int y = 0; y < 28; ++y) {
                    for (int x = 0; x < 28; ++x) {
                        float pixel_value = test_data.m_input[y * 28 + x];
                        if (pixel_value > 0.8f) {
                            std::cout << "##";
                        } else if (pixel_value > 0.5f) {
                            std::cout << "**";
                        } else if (pixel_value > 0.2f) {
                            std::cout << "--";
                        } else {
                            std::cout << " ";
                        }
                    }
                    std::cout << std::endl;
                }

                auto label = std::max_element(test_data.m_desired_output.begin(), test_data.m_desired_output.end()) - test_data.m_desired_output.begin();

                auto result = m_selected_device->Evaluate(*network_on_device->second, test_data.m_input);
                auto guessed_number = std::max_element(result.begin(), result.end()) - result.begin();

                std::cout << std::endl << "Label: " << label << std::endl;
                std::cout << "Network output: " << guessed_number;

            } else {
                std::cout << "Input out of range (" << m_test_data.size() << ")";
            }

            return false;
        };

        m_commands["test"].m_description = "Test on the 10k test dataset";
        m_commands["test"].m_handler = [this](const std::vector<std::string>& args) {
            size_t good_answers = 0;

            auto network_on_device = m_uploaded_networks.find(m_selected_device);
            if (network_on_device == m_uploaded_networks.end()) {
                m_uploaded_networks[m_selected_device] = m_selected_device->RegisterNetwork(*m_network);
                network_on_device = m_uploaded_networks.find(m_selected_device);
            }

            for (size_t i = 0; i < m_test_data.size(); ++i) {
                auto result = m_selected_device->Evaluate(*network_on_device->second, m_test_data[i].m_input);
                auto guessed_number = std::max_element(result.begin(), result.end()) - result.begin();
                auto reference_solution = std::max_element(m_test_data[i].m_desired_output.begin(), m_test_data[i].m_desired_output.end()) - m_test_data[i].m_desired_output.begin();
                if (guessed_number == reference_solution) {
                    ++good_answers;
                }
            }

            std::cout << "Test dataset count: " << m_test_data.size() << std::endl;
            std::cout << "Good answers: " << good_answers << std::endl;
            std::cout << "Result: " << (float(good_answers) / m_test_data.size()) * 100.0f << "%" << std::endl;

            return false;
        };
    }
};

int main()
{
    MnistTrainerApp app{"D:/Dev/macademy_datasets/mnist_digits"};

    app.Run();
    return 0;
}