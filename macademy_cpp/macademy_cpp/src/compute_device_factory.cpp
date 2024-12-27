#include "compute_device_factory.h"
#include "i_compute_device.h"

#include "cpu_backend/cpu_compute_backend.h"

#ifdef MACADEMY_OPENCL_BACKEND
#include "opencl_backend/opencl_compute_device.h"
#endif

#ifdef MACADEMY_VULKAN_BACKEND
#include "vulkan_backend/vulkan_compute_device.h"
#endif

#include "common.h"
#include "utils.h"

namespace macademy::ComputeDeviceFactory {

std::vector<ComputeDeviceInfo> EnumerateComputeDevices()
{
    std::vector<ComputeDeviceInfo> ret;

    ret.emplace_back(CPUComputeDevice::GetCpuComputeDeviceInfo());

#ifdef MACADEMY_OPENCL_BACKEND
    auto opencl_devices = OpenCLComputeDevice::GetOpenCLComputeDeviceInfo();
    std::copy(opencl_devices.begin(), opencl_devices.end(), std::back_inserter(ret));
#endif

#ifdef MACADEMY_VULKAN_BACKEND
    auto vulkan_devices = VulkanComputeDevice::GetVulkanComputeDeviceInfo();
    std::copy(vulkan_devices.begin(), vulkan_devices.end(), std::back_inserter(ret));
#endif

    return ret;
}

std::unique_ptr<IComputeDevice> CreateComputeDevice(const ComputeDeviceInfo& compute_device_info)
{
    if (compute_device_info.m_backend == "cpu") {
        return std::make_unique<CPUComputeDevice>();
    }

#ifdef MACADEMY_OPENCL_BACKEND
    if (compute_device_info.m_backend == "opencl") {
        return std::make_unique<OpenCLComputeDevice>(compute_device_info);
    }
#endif

#ifdef MACADEMY_VULKAN_BACKEND
    if (compute_device_info.m_backend == "vulkan") {
        return std::make_unique<VulkanComputeDevice>(compute_device_info);
    }
#endif

    throw std::runtime_error("Invalid device info!");
}

} // namespace macademy::ComputeDeviceFactory