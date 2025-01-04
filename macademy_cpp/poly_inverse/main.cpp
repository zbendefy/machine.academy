#include "macademy_utils/console_app.h"
#include "utils.h"

#include <fstream>
#include <iostream>
#include <random>
#include <limits>

using namespace std::chrono_literals;
using namespace macademy;

//Trains a neural network to fit a polynom to an input dataset
//Input layer of the network: Uniformly distributed dataset
//Output layer of the network: The polynom coefficients


class PolyFitterApp : public ConsoleApp
{
    const uint32_t poly_rank = 3; //Rank of the polynoms to learn to fit
    const int resolution = 100; //The number of input points
    const float range_min = 0.0f; //The input range (minimum value of x to consider)
    const float range_max = 2.06941f; //The input range (maximum value of x to consider)
    const float poly_coefficient_range = 200.0f; //The min/max number of the polynom coefficients to consider
    const float poly_value_range = 4000.0f; //The min/max number of the polynom value to consider

    Training m_trainer;
    std::shared_ptr<TrainingSuite> m_training_suite;
    std::vector<TrainingData> m_test_data;

    const float _full_poly_coefficient_range_pack_factor = 1.0f / (poly_coefficient_range * 2.0f);
    const float _full_poly_value_range_pack_factor = 1.0f / (poly_value_range * 2.0f);
    const float _full_poly_coefficient_range_unpack_factor = poly_coefficient_range * 2.0f;
    const float _full_poly_value_range_unpack_factor = poly_value_range * 2.0f;

    static float EvalPolynom(std::span<float> coefficients, float x)
    {
        float r_i = 1.0f;
        float val = 0.0f;
        
        for (int i = 0; i < coefficients.size(); ++i)
        {
            val += r_i * coefficients[i];
            r_i *= x;
        }

        return val;
    }

    inline float PackPolyCoefficient(float c)
    {
        return (c + poly_coefficient_range) * _full_poly_coefficient_range_pack_factor;
    }

    inline float PackPolyValue(float v)
    {
        return (std::clamp(v, -poly_value_range, poly_value_range) + poly_value_range) * _full_poly_value_range_pack_factor;
    }

    inline float UnpackPolyCoefficient(float c)
    {
        return c * _full_poly_coefficient_range_unpack_factor - poly_coefficient_range;
    }

    inline float UnpackPolyValue(float v)
    {
        return v * _full_poly_value_range_unpack_factor - poly_value_range;
    }
    void GeneratePolyData(std::vector<TrainingData>& training_data_vector, uint32_t count, uint32_t random_seed)
    {
        std::vector<float> coefficients;
        coefficients.resize(poly_rank);

        std::random_device                  rand_dev;
        std::mt19937                        generator(rand_dev());
        generator.seed(random_seed);
        std::uniform_real_distribution<float>  distr(-poly_coefficient_range, poly_coefficient_range);

        training_data_vector.resize(count);

        for (uint32_t i = 0; i < count; ++i)
        {
            auto& input = training_data_vector[i].m_input;
            input.resize(resolution);

            auto& desired_output = training_data_vector[i].m_desired_output;
            desired_output.resize(poly_rank);


            for (uint32_t c = 0; c < poly_rank; ++c)
            {
                coefficients[c] = distr(generator);

                desired_output[c] = PackPolyCoefficient(coefficients[c]);
            }

            const float step = (range_max - range_min) / (resolution - 1);
            for (uint32_t c = 0; c < resolution; ++c)
            {
                const float x = range_min + c * step;
                const float y = EvalPolynom(coefficients, x);
                input[c] = PackPolyValue(y);
            }
        }
    }

  public:
    PolyFitterApp(const std::string& data_folder)
    {
        std::vector<macademy::LayerConfig> layers;
        layers.emplace_back(macademy::LayerConfig{ .m_activation = macademy::ActivationFunction::LeakyReLU, .m_num_neurons = 32 });
        layers.emplace_back(macademy::LayerConfig{ .m_activation = macademy::ActivationFunction::LeakyReLU, .m_num_neurons = 32 });
        layers.emplace_back(macademy::LayerConfig{ .m_activation = macademy::ActivationFunction::LeakyReLU, .m_num_neurons = 32 });
        layers.emplace_back(macademy::LayerConfig{.m_activation = macademy::ActivationFunction::LeakyReLU, .m_num_neurons = poly_rank});
        m_network = macademy::NetworkFactory::Build("Polynom fitter", resolution, std::span<macademy::LayerConfig>(layers.data(), layers.size()), DataType::Float32);

        m_network->GenerateRandomWeights(macademy::XavierWeightInitializer{});

        m_training_suite = std::make_shared<TrainingSuite>();
        m_training_suite->m_mini_batch_size = 200;
        m_training_suite->m_cost_function = CostFunction::MeanSquared;
        m_training_suite->m_regularization = Regularization::L2;
        m_training_suite->m_learning_rate = 0.1f;
        m_training_suite->m_shuffle_training_data = true;

        GeneratePolyData(m_training_suite->m_training_data, /*count =*/ 100000, /*seed = */1234);
        
        GeneratePolyData(m_test_data, /*count =*/ 10000, /*seed = */3213);

        m_commands["train"].m_description = "Train the network";
        m_commands["train"].m_handler = [this](const std::vector<std::string>& args) {
            m_break_requested = false;
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

            auto time_begin = std::chrono::high_resolution_clock::now();

            auto tracker = m_trainer.Train(*network_on_device->second, *m_selected_device, m_training_suite);

            std::cout << std::endl;

            TrainingDisplay(*tracker);

            auto time_end = std::chrono::high_resolution_clock::now();
            auto duration = duration_cast<std::chrono::milliseconds>(time_end - time_begin);

            std::cout << "Training time: " << duration << std::endl;

            return false;
        };

        m_commands["test"].m_description = "Test on the test dataset";
        m_commands["test"].m_handler = [this](const std::vector<std::string>& args) {
            auto network_on_device = m_uploaded_networks.find(m_selected_device);
            if (network_on_device == m_uploaded_networks.end()) {
                m_uploaded_networks[m_selected_device] = m_selected_device->RegisterNetwork(*m_network);
                network_on_device = m_uploaded_networks.find(m_selected_device);
            }

            auto [total_avg_error, min_error, max_error] = TestNetwork(*network_on_device->second);

            std::cout << "Test dataset count: " << m_test_data.size() << std::endl;
            std::cout << "Average error per test polinom: " << total_avg_error << std::endl;
            std::cout << "Min error: " << min_error << std::endl;
            std::cout << "Max error: " << max_error << std::endl;

            return false;
        };
    }

    std::tuple<double, double, double> TestNetwork(const NetworkResourceHandle& network)
    {
        double total_avg_error = 0;
        double min_error = std::numeric_limits<float>::max();
        double max_error = std::numeric_limits<float>::min();

        for (size_t i = 0; i < m_test_data.size(); ++i) {
            double error = 0;
            auto result = m_selected_device->Evaluate(network, m_test_data[i].m_input);
            for (uint32_t c = 0; c < poly_rank; ++c)
            {
                error += std::abs(UnpackPolyCoefficient(result[c]) - UnpackPolyCoefficient(m_test_data[i].m_desired_output[c]));
            }
            total_avg_error += error;
            min_error = std::min(error, min_error);
            max_error = std::max(error, max_error);
        }

        total_avg_error /= m_test_data.size();

        return { total_avg_error, min_error, max_error };
    }
};


#ifdef _WIN32
PolyFitterApp* __app_ptr = nullptr;
#include <Windows.h>
#undef max
BOOL WINAPI ConsoleHandlerRoutine(DWORD dwCtrlType)
{
    if (dwCtrlType == CTRL_C_EVENT || dwCtrlType == CTRL_BREAK_EVENT)
    {
        __app_ptr->OnBreak();
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        return TRUE;
    }

    return FALSE;
}
#endif

int main()
{
    PolyFitterApp app{"D:/Dev/macademy_datasets/mnist_digits"};

#ifdef _WIN32
    __app_ptr = &app;

    if (!SetConsoleCtrlHandler(ConsoleHandlerRoutine, TRUE))
    {
        std::cerr << "Failed to register break handler!" << std::endl;
    }
#endif

    app.Run();
    return 0;
}