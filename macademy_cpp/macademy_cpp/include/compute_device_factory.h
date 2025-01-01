#pragma once

#include <vector>
#include <memory>
#include <nlohmann/json.hpp>

namespace macademy {

struct ComputeDeviceInfo;
class IComputeDevice;

namespace ComputeDeviceFactory {

std::vector<ComputeDeviceInfo> EnumerateComputeDevices();

std::unique_ptr<IComputeDevice> CreateComputeDevice(const ComputeDeviceInfo& compute_device_info, const nlohmann::json& device_config = {});

} // namespace ComputeDeviceFactory

} // namespace macademy