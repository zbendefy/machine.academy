#include "macademy_utils/console_app.h"
#include "utils.h"

#include <fstream>
#include <iostream>

using namespace std::chrono_literals;
using namespace macademy;

class MandelbrotTrainerApp : public ConsoleApp
{
    static const int img_dimension = 28;

    Training m_trainer;
    std::shared_ptr<TrainingSuite> m_training_suite;
    std::vector<TrainingData> m_test_data;

  public:
    MandelbrotTrainerApp()
    {
        std::vector<macademy::LayerConfig> layers;
        layers.emplace_back(macademy::LayerConfig{.m_activation = macademy::ActivationFunction::Sigmoid, .m_num_neurons = 24});
        layers.emplace_back(macademy::LayerConfig{.m_activation = macademy::ActivationFunction::Sigmoid, .m_num_neurons = 10});
        m_network = macademy::NetworkFactory::Build("Mandelbrot drawer", img_dimension * img_dimension, std::span<macademy::LayerConfig>(layers.data(), layers.size()));

        m_network->GenerateRandomWeights(macademy::XavierWeightInitializer{});

        m_training_suite = std::make_shared<TrainingSuite>();
        m_training_suite->m_mini_batch_size = 100;
        m_training_suite->m_cost_function = CostFunction::CrossEntropy_Sigmoid;
        m_training_suite->m_regularization = Regularization::L2;
        m_training_suite->m_learning_rate = 0.005f;
        m_training_suite->m_shuffle_training_data = true;

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

            EnsureNetworkResources();

            m_training_suite->m_epochs = epochs;

            auto time_begin = std::chrono::high_resolution_clock::now();

            auto tracker = m_trainer.Train(*m_network_resources, m_training_suite);

            std::cout << std::endl;

            TrainingDisplay(*tracker);

            auto time_end = std::chrono::high_resolution_clock::now();
            auto duration = duration_cast<std::chrono::milliseconds>(time_end - time_begin);

            std::cout << "Training time: " << duration.count() << "ms" << std::endl;

            return false;
        };

        m_commands["eval"].m_description = "Eval the test input [n] on the network";
        m_commands["eval"].m_handler = [this](const std::vector<std::string>& args) {
            int input = 0;
            bool eval_from_training_dataset = false;
            bool verbose = false;
            for (int i = 1; i < args.size(); ++i) {
                if (args[i] == "training") {
                    eval_from_training_dataset = true;
                }
                if (args[i] == "verbose") {
                    verbose = true;
                }

                switch (i) {
                case 1:
                    input = atof(args[i].c_str());
                    break;
                }
            }

            EnsureNetworkResources();

            std::vector<TrainingData>* dataset = nullptr;

            if (eval_from_training_dataset) {
                dataset = &m_training_suite->m_training_data;
                std::cout << "Eval from training dataset, #" << input << " of " << m_training_suite->m_training_data.size() << std::endl;
            } else {
                dataset = &m_test_data;
                std::cout << "Eval from test dataset, #" << input << " of " << m_test_data.size() << std::endl;
            }

            if (input < dataset->size() && input >= 0) {
                auto& test_data = (*dataset)[input];
                for (int y = 0; y < img_dimension; ++y) {
                    for (int x = 0; x < img_dimension; ++x) {
                        float pixel_value = test_data.m_input[y * img_dimension + x];
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

                auto result = m_compute_tasks.Evaluate(*m_network_resources, test_data.m_input);
                auto guessed_number = std::max_element(result.begin(), result.end()) - result.begin();

                std::cout << std::endl << "Label: " << label << std::endl;
                std::cout << "Network output: " << guessed_number << std::endl;

                if (verbose) {
                    std::cout << "Raw network output: " << std::endl;
                    for (auto o : result) {
                        std::cout << o << std::endl;
                    }
                }

            } else {
                std::cout << "Input out of range (" << dataset->size() << ")";
            }

            return false;
        };

        m_commands["test"].m_description = "Test on the 10k test dataset";
        m_commands["test"].m_handler = [this](const std::vector<std::string>& args) {
            EnsureNetworkResources();

            size_t good_answers = TestNetwork(*m_network_resources);

            std::cout << "Test dataset count: " << m_test_data.size() << std::endl;
            std::cout << "Good answers: " << good_answers << std::endl;
            std::cout << "Result: " << (float(good_answers) / m_test_data.size()) * 100.0f << "%" << std::endl;

            return false;
        };
    }

    size_t TestNetwork(const NetworkResourceHandle& network)
    {
        size_t good_answers = 0;
        for (size_t i = 0; i < m_test_data.size(); ++i) {
            auto result = m_compute_tasks.Evaluate(network, m_test_data[i].m_input);
            auto guessed_number = std::max_element(result.begin(), result.end()) - result.begin();
            auto reference_solution = std::max_element(m_test_data[i].m_desired_output.begin(), m_test_data[i].m_desired_output.end()) - m_test_data[i].m_desired_output.begin();
            if (guessed_number == reference_solution) {
                ++good_answers;
            }
        }

        return good_answers;
    }
};

int main()
{
    MandelbrotTrainerApp app{};

    app.Run();
    return 0;
}