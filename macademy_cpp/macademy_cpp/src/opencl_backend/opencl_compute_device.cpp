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
    {
        
    }

    std::unique_ptr<NetworkResourceHandle> OpenCLComputeDevice::RegisterNetwork(Network& network)
    {
        return std::make_unique<OpenCLNetworkResourceHandle>(&network);
    }

    std::vector<float> OpenCLComputeDevice::Evaluate(const NetworkResourceHandle& network_handle, const std::span<float>& input) const
    {
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