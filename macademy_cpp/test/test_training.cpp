#include <gtest/gtest.h>

#include "network.h"
#include "default_weight_initializer.h"
#include "training_suite.h"
#include "cpu_backend/cpu_compute_backend.h"
#ifdef MACADEMY_OPENCL_BACKEND
#include "opencl_backend/opencl_compute_device.h"
#endif
#ifdef MACADEMY_VULKAN_BACKEND
#include "vulkan_backend/vulkan_compute_device.h"
#endif
#include "compute_device_factory.h"
#include "compute_tasks.h"
#include "utils.h"
#include <span>

using namespace macademy;

class TrainingTest : public ::testing::Test
{
  public:
    ComputeTasks m_compute_tasks;

    TrainingTest() {}

    void RunApplyGradientTest(const ComputeDeviceInfo& device_info)
    {
        std::vector<LayerConfig> layers;
        layers.emplace_back(LayerConfig{.m_activation = ActivationFunction::Sigmoid, .m_num_neurons = 24});
        layers.emplace_back(LayerConfig{.m_activation = ActivationFunction::Sigmoid, .m_num_neurons = 13});
        layers.emplace_back(LayerConfig{.m_activation = ActivationFunction::Sigmoid, .m_num_neurons = 24});
        layers.emplace_back(LayerConfig{.m_activation = ActivationFunction::Sigmoid, .m_num_neurons = 32});

        const auto network = NetworkFactory::Build("test", 5, std::span<const LayerConfig>(layers.data(), layers.size()));
        network->GenerateRandomWeights(macademy::XavierWeightInitializer{});

        auto compute_device = ComputeDeviceFactory::CreateComputeDevice(device_info);
        auto network_resources = std::make_unique<NetworkResourceHandle>(*network, *compute_device);

        std::vector<float> orig_weights;
        orig_weights.resize(network->GetRawWeightData().size());
        std::copy(network->GetRawWeightData().begin(), network->GetRawWeightData().end(), orig_weights.begin());

        std::vector<float> gradient;
        gradient.resize(network->GetRawWeightData().size());
        for (auto& g : gradient) {
            g = 2.53f;
        }

        // TODO run apply gradients
    }

    void RunTrainingTest(const ComputeDeviceInfo& device_info)
    {
        constexpr int input_output_size = 4;

        constexpr uint32_t minibatch_size = 3;

        std::vector<LayerConfig> layers;
        layers.emplace_back(LayerConfig{.m_activation = ActivationFunction::Sigmoid, .m_num_neurons = input_output_size});
        layers.emplace_back(LayerConfig{.m_activation = ActivationFunction::Sigmoid, .m_num_neurons = input_output_size});

        const auto network = NetworkFactory::Build("test", input_output_size, std::span<const LayerConfig>(layers.data(), layers.size()));

        auto compute_device = ComputeDeviceFactory::CreateComputeDevice(device_info);
        auto network_resources = std::make_unique<NetworkResourceHandle>(*network, *compute_device);

        TrainingSuite ts{};
        ts.m_cost_function = CostFunction::CrossEntropy_Sigmoid;
        ts.m_epochs = 20000;
        ts.m_learning_rate = 0.01f;
        ts.m_mini_batch_size = minibatch_size;
        ts.m_regularization = Regularization::L2;
        ts.m_shuffle_training_data = true;

        std::vector<float> tmp;
        tmp.resize(input_output_size, 0.0f);

        std::random_device seed;
        std::mt19937 gen{seed()};                                       // seed the generator
        std::uniform_int_distribution<> dist{0, input_output_size - 1}; // set min and max

        for (uint32_t i = 0; i < 1000; ++i) {
            TrainingData td;
            td.m_desired_output = tmp;
            td.m_input = tmp;
            const auto val = dist(gen);
            td.m_desired_output[val] = 1.0f;
            td.m_input[val] = 1.0f;
            ts.m_training_data.push_back(std::move(td));
        }

        network_resources->AllocateTrainingResources(ts.m_mini_batch_size ? *ts.m_mini_batch_size : ts.m_training_data.size());
        for (int i = 0; i < ts.m_epochs; ++i) {
            int training_data_idx = 0;
            while (training_data_idx < uint32_t(ts.m_training_data.size())) {
                m_compute_tasks.TrainMinibatch(*network_resources, ts, training_data_idx, std::min(training_data_idx + minibatch_size, uint32_t(ts.m_training_data.size())));
                training_data_idx += minibatch_size;
            }
        }

        for (int i = 0; i < input_output_size; ++i) {
            printf("testing %d...\n", i);
            auto test = tmp;
            test[i] = 1.0f;
            auto output = m_compute_tasks.Evaluate(*network_resources, test);

            for (uint32_t k = 0; k < uint32_t(output.size()); ++k) {
                if (k == i) {
                    EXPECT_GT(output[k], 0.8f);
                } else {
                    EXPECT_LT(output[k], 0.2f);
                }
            }
        }
    }
};

TEST_F(TrainingTest, Training)
{
    auto cpu_compute_device_info = CPUComputeDevice::GetCpuComputeDeviceInfo();
    RunTrainingTest(cpu_compute_device_info);
}
