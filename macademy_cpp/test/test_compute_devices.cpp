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
        layers.emplace_back(LayerConfig{.m_activation = ActivationFunction::Sigmoid, .m_num_neurons = 4});
        layers.emplace_back(LayerConfig{.m_activation = ActivationFunction::ReLU, .m_num_neurons = 15});
        layers.emplace_back(LayerConfig{.m_activation = ActivationFunction::Sigmoid, .m_num_neurons = 2});
        layers.emplace_back(LayerConfig{.m_activation = ActivationFunction::Sigmoid, .m_num_neurons = 2003});
        layers.emplace_back(LayerConfig{.m_activation = ActivationFunction::Sigmoid, .m_num_neurons = 2048});
        layers.emplace_back(LayerConfig{.m_activation = ActivationFunction::Sigmoid, .m_num_neurons = 8});
        m_network = NetworkFactory::Build("test", 5, std::span<LayerConfig>(layers.data(), layers.size()));

        m_network->GenerateRandomWeights(XavierWeightInitializer{});
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

    void TestComputeDeviceBatchEvalToReference(std::span<ComputeDeviceInfo> devices)
    {
        std::vector<float> input{
            1,      -2,    3,    -10,  10,  // input1
            -3,     1,     2,    -5,   4,   // input2
            0.5f,   -1,    3,    2,    -1,  // input3
            -7.41f, 1.23f, 1.3f, 3.4f, 7.8f // input4
        };

        auto cpu_compute_device_info = CPUComputeDevice::GetCpuComputeDeviceInfo();
        auto cpu_compute_device = ComputeDeviceFactory::CreateComputeDevice(cpu_compute_device_info);
        auto cpu_network_resources = std::make_unique<NetworkResourceHandle>(*m_network, *cpu_compute_device);

        auto reference_results = m_compute_tasks.EvaluateBatch(4, *cpu_network_resources, input);

        for (auto device_info : devices) {
            std::cout << "Testing Device #" << device_info.m_device_index << " - " << device_info.m_device_name << std::endl;
            auto compute_device = ComputeDeviceFactory::CreateComputeDevice(device_info);
            auto network_resources = std::make_unique<NetworkResourceHandle>(*m_network, *compute_device);
            auto result = m_compute_tasks.EvaluateBatch(4, *network_resources, input);

            ASSERT_EQ(reference_results.size(), result.size());
            for (size_t i = 0; i < result.size(); ++i) {
                EXPECT_NEAR(reference_results[i], result[i], 1e-3);
            }
        }
    }

    void TestSingleVsBatchEval(const ComputeDeviceInfo& device_info)
    {
        // Tests if 4 single inference evaluation runs produce the same exact results as with a single batched run

        std::vector<float> input1{1, -2, 3, -10, 10};
        std::vector<float> input2{-3, 1, 2, -5, 4};
        std::vector<float> input3{0.5f, -1, 3, 2, -1};
        std::vector<float> input4{-7.41f, 1.23f, 1.3f, 3.4f, 7.8f};

        std::vector<float> batched_input{};
        std::copy(input1.begin(), input1.end(), std::back_inserter(batched_input));
        std::copy(input2.begin(), input2.end(), std::back_inserter(batched_input));
        std::copy(input3.begin(), input3.end(), std::back_inserter(batched_input));
        std::copy(input4.begin(), input4.end(), std::back_inserter(batched_input));

        auto compute_device = ComputeDeviceFactory::CreateComputeDevice(device_info);
        auto network_resources = std::make_unique<NetworkResourceHandle>(*m_network, *compute_device);

        auto result1 = m_compute_tasks.Evaluate(*network_resources, input1); // result1 is already checked by a previous test
        auto result2 = m_compute_tasks.Evaluate(*network_resources, input2);
        auto result3 = m_compute_tasks.Evaluate(*network_resources, input3);
        auto result4 = m_compute_tasks.Evaluate(*network_resources, input4);

        std::vector<float> batched_reference_results{};
        std::copy(result1.begin(), result1.end(), std::back_inserter(batched_reference_results));
        std::copy(result2.begin(), result2.end(), std::back_inserter(batched_reference_results));
        std::copy(result3.begin(), result3.end(), std::back_inserter(batched_reference_results));
        std::copy(result4.begin(), result4.end(), std::back_inserter(batched_reference_results));

        auto batched_results = m_compute_tasks.EvaluateBatch(4, *network_resources, batched_input);

        ASSERT_EQ(batched_reference_results.size(), batched_results.size());
        for (size_t i = 0; i < batched_results.size(); ++i) {
            EXPECT_FLOAT_EQ(batched_reference_results[i], batched_results[i]);
        }
    }

    void TestComputeDeviceBatchDeterministicCheck(const ComputeDeviceInfo& device_info)
    {
        // Checks if multiple runs produce the same exact results

        std::vector<float> input{
            1,      -2,    3,    -10,  10,  // input1
            -3,     1,     2,    -5,   4,   // input2
            0.5f,   -1,    3,    2,    -1,  // input3
            -7.41f, 1.23f, 1.3f, 3.4f, 7.8f // input4
        };

        constexpr uint32_t num_runs = 5;

        std::vector<float> reference_results;

        for (uint32_t d = 0; d < num_runs; ++d) {
            std::cout << "Testing Device #" << device_info.m_device_index << " - " << device_info.m_device_name << std::endl;
            auto compute_device = ComputeDeviceFactory::CreateComputeDevice(device_info);
            auto network_resources = std::make_unique<NetworkResourceHandle>(*m_network, *compute_device);
            auto result = m_compute_tasks.EvaluateBatch(4, *network_resources, input);

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
};

TEST_F(ComputeDevicesTest, Utils) { EXPECT_EQ(2048, CalculateLargestLayerNeuronCount(m_network->GetLayerConfig())); }

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

TEST_F(ComputeDevicesTest, CPUComputeDeviceBatchEval) { TestSingleVsBatchEval(CPUComputeDevice::GetCpuComputeDeviceInfo()); }

TEST_F(ComputeDevicesTest, CPUComputeDeviceDeterministicCheck) { TestComputeDeviceBatchDeterministicCheck(CPUComputeDevice::GetCpuComputeDeviceInfo()); }

#ifdef MACADEMY_OPENCL_BACKEND
TEST_F(ComputeDevicesTest, OpenCLComputeDevice)
{
    auto opencl_devices = OpenCLComputeDevice::GetOpenCLComputeDeviceInfo();

    TestComputeDeviceToReference(opencl_devices);
}

TEST_F(ComputeDevicesTest, OpenCLComputeDeviceBatchEval)
{
    auto opencl_devices = OpenCLComputeDevice::GetOpenCLComputeDeviceInfo();

    for (const auto& it : opencl_devices) {
        TestSingleVsBatchEval(it);
    }
}

TEST_F(ComputeDevicesTest, OpenCLComputeDeviceBatchEvalToReference)
{
    auto opencl_devices = OpenCLComputeDevice::GetOpenCLComputeDeviceInfo();

    TestComputeDeviceBatchEvalToReference(opencl_devices);
}

TEST_F(ComputeDevicesTest, OpenCLComputeDeviceDeterministickCheck)
{
    auto opencl_devices = OpenCLComputeDevice::GetOpenCLComputeDeviceInfo();

    for (const auto& it : opencl_devices) {
        TestComputeDeviceBatchDeterministicCheck(it);
    }
}
#endif

#ifdef MACADEMY_VULKAN_BACKEND
TEST_F(ComputeDevicesTest, VulkanComputeDevice)
{
    auto vk_devices = VulkanComputeDevice::GetVulkanComputeDeviceInfo();

    TestComputeDeviceToReference(vk_devices);
}

TEST_F(ComputeDevicesTest, VulkanComputeDeviceBatchEval)
{
    auto vk_devices = VulkanComputeDevice::GetVulkanComputeDeviceInfo();

    for (const auto& it : vk_devices) {
        TestSingleVsBatchEval(it);
    }
}

TEST_F(ComputeDevicesTest, VulkanComputeDeviceBatchEvalToReference)
{
    auto vk_devices = VulkanComputeDevice::GetVulkanComputeDeviceInfo();

    TestComputeDeviceBatchEvalToReference(vk_devices);
}

TEST_F(ComputeDevicesTest, VulkanComputeDeviceDeterministickCheck)
{
    auto vk_devices = VulkanComputeDevice::GetVulkanComputeDeviceInfo();

    for (const auto& it : vk_devices) {
        TestComputeDeviceBatchDeterministicCheck(it);
    }
}

#endif