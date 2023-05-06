#pragma once

#include "i_compute_backend.h"
#include "opencl_common.h"

#include <optional>

namespace macademy {

struct OpenCLDeviceConfig
{
    bool m_mad_enable = true;
    bool m_fast_relaxed_math = true;
    bool m_no_signed_zeros = true;
    bool m_unsafe_math_optimizations = false;
    std::optional<uint32_t> m_optimal_threadgroup_size;
};

class OpenCLComputeDevice : public IComputeDevice
{
    cl::Device m_device;
    OpenCLDeviceConfig m_device_config;
    cl::Context m_context;
    mutable cl::CommandQueue m_command_queue;
    cl::Program m_program;

    mutable std::unique_ptr<cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl_uint, cl_ulong>> m_kernel_calc_single_layer;
    cl::size_type m_kernel_calc_single_layer_ideal_workgroup_size = 32;

  public:
    OpenCLComputeDevice(cl::Device device, OpenCLDeviceConfig advanced_config = {});

    virtual std::unique_ptr<NetworkResourceHandle> RegisterNetwork(Network& network) override;

    virtual std::vector<float> Evaluate(const NetworkResourceHandle& network_handle, std::span<const float> input) const override;

    virtual void Train(const NetworkResourceHandle& network, const TrainingSuite& training_suite, uint32_t trainingDataBegin, uint32_t trainingDataEnd) const override;

    static std::vector<cl::Device> GetDeviceList();

    static cl::Device AutoSelectDevice();

    std::string GetDeviceName() const override;

    size_t GetTotalMemory() const override;

    uint32_t GetComputeUnits() const override;
};
} // namespace macademy