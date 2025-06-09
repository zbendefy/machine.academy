#include <gtest/gtest.h>

#include "network.h"
#include "default_weight_initializer.h"
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

using namespace macademy;

class ComputeDevicesTest : public ::testing::Test
{
  public:
    std::unique_ptr<Network> m_network;
    ComputeTasks m_compute_tasks;

    ComputeDevicesTest()
    {
        std::vector<LayerConfig> layers;
        layers.emplace_back(LayerConfig{.m_activation_function = ActivationFunction::Sigmoid, .m_num_neurons = 4});
        layers.emplace_back(LayerConfig{.m_activation_function = ActivationFunction::ReLU, .m_num_neurons = 15});
        layers.emplace_back(LayerConfig{.m_activation_function = ActivationFunction::Sigmoid, .m_num_neurons = 2});
        layers.emplace_back(LayerConfig{.m_activation_function = ActivationFunction::Sigmoid, .m_num_neurons = 2003});
        layers.emplace_back(LayerConfig{.m_activation_function = ActivationFunction::Sigmoid, .m_num_neurons = 2048});
        layers.emplace_back(LayerConfig{.m_activation_function = ActivationFunction::Sigmoid, .m_num_neurons = 8});
        m_network = BuildSequentialNetwork("test", 5, std::span<LayerConfig>(layers.data(), layers.size()), XavierWeightInitializer{});
    }

    void TestComputeDeviceToReference(std::span<ComputeDeviceInfo> devices)
    {
        std::vector<float> input{1, -2, 3, -10, 10};

        auto cpu_compute_device_info = CPUComputeDevice::GetCpuComputeDeviceInfo();
        auto cpu_compute_device = ComputeDeviceFactory::CreateComputeDevice(cpu_compute_device_info);
        auto cpu_network_resources = std::make_unique<NetworkResourceHandle>(*m_network, *cpu_compute_device);
        auto reference_results = m_compute_tasks.Evaluate(*cpu_network_resources, input);

        for (auto device_info : devices) {
            std::cout << "Testing Device #" << device_info.m_device_index << " - " << device_info.m_device_name << std::endl;
            auto compute_device = ComputeDeviceFactory::CreateComputeDevice(device_info);
            auto network_resources = std::make_unique<NetworkResourceHandle>(*m_network, *compute_device);
            auto result = m_compute_tasks.Evaluate(*network_resources, input);

            ASSERT_EQ(reference_results.size(), result.size());
            for (size_t i = 0; i < result.size(); ++i) {
                EXPECT_NEAR(reference_results[i], result[i], 1e-3);
            }
        }
    }

    void TestComputeDeviceDeterministicCheck(const ComputeDeviceInfo& device_info)
    {
        // Checks if multiple runs produce the same exact results

        std::vector<float> input{
            1,      -2,    3,    -10,  10
        };

        constexpr uint32_t num_runs = 5;

        std::vector<float> reference_results;

        for (uint32_t d = 0; d < num_runs; ++d) {
            std::cout << "Testing Device #" << device_info.m_device_index << " - " << device_info.m_device_name << std::endl;
            auto compute_device = ComputeDeviceFactory::CreateComputeDevice(device_info);
            auto network_resources = std::make_unique<NetworkResourceHandle>(*m_network, *compute_device);
            auto result = m_compute_tasks.Evaluate(*network_resources, input);

            if (d == 0) {
                reference_results = result;
            } else {
                ASSERT_EQ(reference_results.size(), result.size());
                for (size_t i = 0; i < result.size(); ++i) {
                    EXPECT_NEAR(reference_results[i], result[i], 1e-3);
                }
            }
        }
    }

    void TestForwardPass(const ComputeDeviceInfo& device_info)
    {
        auto reference_device = ComputeDeviceFactory::CreateComputeDevice(CPUComputeDevice::GetCpuComputeDeviceInfo());
        auto compute_device = ComputeDeviceFactory::CreateComputeDevice(device_info);

        auto test_device = [](IComputeDevice& compute_device) {
            const uint32_t prev_layer_num_neurons = 5;
            const uint32_t num_neurons = 10;
            const uint32_t num_weights = (prev_layer_num_neurons + 1) * num_neurons;
            const uint32_t num_training_samples = 5;

            auto tensor_buffer = compute_device.CreateBuffer(num_weights * sizeof(float), BufferUsage::ReadWrite, "tensor");
            auto prev_activations_buffer = compute_device.CreateBuffer(num_training_samples * prev_layer_num_neurons * sizeof(float), BufferUsage::ReadWrite, "prev_activations");
            auto activations_buffer = compute_device.CreateBuffer(num_training_samples * num_neurons * sizeof(float), BufferUsage::ReadWrite, "activations");
            auto zvalues_buffer = compute_device.CreateBuffer(num_training_samples * num_neurons * sizeof(float), BufferUsage::ReadWrite, "zvalues");

            std::vector<float> weights{};
            for (int i = 0; i < num_weights; ++i)
            {
                weights.emplace_back(fmod(weights.size() * 13412.3231341f, 2.5213f) - 0.0356f * weights.size()-1.2421f);
            }
            std::vector<float> prev_activations{};
            for (int i = 0; i < num_training_samples * prev_layer_num_neurons; ++i)
            {
                prev_activations.emplace_back(fmod(prev_activations.size() * 1342.3231341f, 1.0f));
            }

            std::vector<float> results_activations, results_zvalues;
            results_activations.resize(num_training_samples * num_neurons);
            results_zvalues.resize(num_training_samples * num_neurons);

            compute_device.QueueWriteToBuffer(tensor_buffer.get(), ToReadOnlyUi8Span(weights), 0);
            compute_device.QueueWriteToBuffer(prev_activations_buffer.get(), ToReadOnlyUi8Span(prev_activations), 0);

            compute_device.QueueTrainForwardPass(tensor_buffer.get(), prev_activations_buffer.get(), activations_buffer.get(), zvalues_buffer.get(), ActivationFunction::Sigmoid, num_neurons, prev_layer_num_neurons, num_training_samples);
            
            compute_device.QueueReadFromBuffer(activations_buffer.get(), ToWriteableUi8Span(results_activations), 0);
            compute_device.QueueReadFromBuffer(zvalues_buffer.get(), ToWriteableUi8Span(results_zvalues), 0);

            compute_device.SubmitQueue();
            compute_device.WaitQueueIdle();
            return std::make_pair(results_activations, results_zvalues);
            };


        auto [reference_activations, reference_zvalues] = test_device(*reference_device);
        auto [test_activations, test_zvalues] = test_device(*compute_device);

        ASSERT_EQ(reference_activations.size(), test_activations.size());
        ASSERT_EQ(reference_activations.size(), test_zvalues.size());
        for (size_t i = 0; i < reference_activations.size(); i++)
        {
            EXPECT_NEAR(reference_activations[i], test_activations[i], 0.0001f);
            EXPECT_NEAR(reference_zvalues[i], test_zvalues[i], 0.0001f);
        }
    }
};

TEST_F(ComputeDevicesTest, Utils) { EXPECT_EQ(2048, CalculateLargestLayerNeuronCount(m_network->GetLayers())); }

TEST_F(ComputeDevicesTest, CPUComputeDevice)
{
    // Checks results to reference values

    std::vector<float> input{1, -2, 3, -10, 10};

    auto cpu_compute_device_info = CPUComputeDevice::GetCpuComputeDeviceInfo();

    auto compute_device = ComputeDeviceFactory::CreateComputeDevice(cpu_compute_device_info);
    auto network_resources = std::make_unique<NetworkResourceHandle>(*m_network, *compute_device);
    auto result = m_compute_tasks.Evaluate(*network_resources, input);

    const std::array<float, 8> reference_result = {
        0.71487468, //
        0.58039027, //
        0.39867547, //
        0.54535365, //
        0.37358889, //
        0.72223479, //
        0.31536984, //
        0.70718151, //
    };

    ASSERT_EQ(reference_result.size(), result.size());
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_FLOAT_EQ(reference_result[i], result[i]);
    }
}

#ifdef MACADEMY_OPENCL_BACKEND
TEST_F(ComputeDevicesTest, OpenCLComputeDevice)
{
    auto opencl_devices = OpenCLComputeDevice::GetOpenCLComputeDeviceInfo();

    TestComputeDeviceToReference(opencl_devices);
}

TEST_F(ComputeDevicesTest, OpenCLComputeDeviceDeterministickCheck)
{
    auto opencl_devices = OpenCLComputeDevice::GetOpenCLComputeDeviceInfo();

    for (const auto& it : opencl_devices) {
        TestComputeDeviceDeterministicCheck(it);
    }
}

TEST_F(ComputeDevicesTest, OpenCLComputeDeviceForwardPassTest)
{
    auto opencl_devices = OpenCLComputeDevice::GetOpenCLComputeDeviceInfo();

    for (const auto& it : opencl_devices) {
        TestForwardPass(it);
    }
}
#endif

#ifdef MACADEMY_VULKAN_BACKEND
TEST_F(ComputeDevicesTest, VulkanComputeDevice)
{
    auto vk_devices = VulkanComputeDevice::GetVulkanComputeDeviceInfo();

    TestComputeDeviceToReference(vk_devices);
}

TEST_F(ComputeDevicesTest, VulkanComputeDeviceDeterministickCheck)
{
    auto vk_devices = VulkanComputeDevice::GetVulkanComputeDeviceInfo();

    for (const auto& it : vk_devices) {
        TestComputeDeviceDeterministicCheck(it);
    }
}

#endif