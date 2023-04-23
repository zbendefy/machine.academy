#include "opencl_backend/opencl_compute_device.h"
#include "network.h"
#include "common.h"

namespace macademy
{
    constexpr const char* calcLayerKernel = "calcSingleLayer";
    constexpr const char* forwardPass = "trainingForwardPass";
    constexpr const char* backwardPassKernel = "trainingBackwardPass";

    struct OpenCLNetworkResourceHandle : public NetworkResourceHandle
    {
        using NetworkResourceHandle::NetworkResourceHandle;
    };

    OpenCLComputeDevice::OpenCLComputeDevice(cl::Device device)
        : m_device(device)
        , m_command_queue()
    {
        std::vector<std::string> programStrings{
            "alma"
        };
        m_program = cl::Program(programStrings);
        m_program.build("");

        m_kernel_calc_single_layer = std::make_unique<cl::KernelFunctor<>>(
            cl::KernelFunctor<>(m_program, "calcSingleLayer"));
    }

    std::unique_ptr<NetworkResourceHandle> OpenCLComputeDevice::RegisterNetwork(Network& network)
    {
        return std::make_unique<OpenCLNetworkResourceHandle>(network);
    }

    std::vector<float> OpenCLComputeDevice::Evaluate(const NetworkResourceHandle& network_handle, const std::span<float>& input) const
    {
        Network& network = *network_handle.m_network;

        if (input.size() != network.GetInputCount())
        {
            throw std::runtime_error("Invalid input length!");
        }

        auto layer_config = network.GetLayerConfig();
        std::vector<float> layer_args = std::vector<float>(input.begin(), input.end());
        std::vector<float> layer_result{};
        float* neuron_weight_data = network.GetRawWeightData().data();
        for (size_t i = 0; i < layer_config.size(); ++i)
        {
            layer_result.clear();
            const uint32_t input_num = uint32_t(layer_args.size());
            const uint32_t output_num = layer_config[i].m_num_neurons;

            layer_result.resize(output_num);

            (*m_kernel_calc_single_layer)(
                cl::EnqueueArgs(
                    cl::NDRange(1)));

            std::swap(layer_args, layer_result);
        }

        ASSERTM(neuron_weight_data - network.GetRawWeightData().data() == network.GetRawWeightData().size(), "");
        return layer_args;


    }

    std::vector<cl::Device> OpenCLComputeDevice::GetDeviceList()
    {
        std::vector<cl::Device> all_devices;
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        for (const auto& platform : platforms)
        {
            std::vector<cl::Device> platform_devices;
            platform.getDevices(CL_DEVICE_TYPE_ALL, &platform_devices);

            std::copy(platform_devices.begin(), platform_devices.end(), std::back_inserter(all_devices));
        }

        return all_devices;
    }
    
    cl::Device OpenCLComputeDevice::AutoSelectDevice()
    {
        auto all_devices = GetDeviceList();
        return all_devices[0];
    }
}