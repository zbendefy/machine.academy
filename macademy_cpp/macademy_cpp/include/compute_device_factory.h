#pragma once

#include <vector>
#include <memory>

namespace macademy {

struct ComputeDeviceInfo;
class IComputeDevice;

namespace ComputeDeviceFactory
{

	std::vector<ComputeDeviceInfo> EnumerateComputeDevices();

	std::unique_ptr<IComputeDevice> CreateComputeDevice(const ComputeDeviceInfo& compute_device_info);

}

}