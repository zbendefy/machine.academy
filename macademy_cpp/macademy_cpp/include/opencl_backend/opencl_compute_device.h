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
    mutable cl::Context m_context;
    mutable cl::CommandQueue m_command_queue;
    cl::Program m_program;

    using KernelEval = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl_uint, cl_ulong>;
    using KernelTrainingForwardPass = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl_uint, cl_ulong, cl_uint, cl_uint>;
    using KernelTrainingBackwardPass =
        cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl_uint, cl_uint, cl_uint, cl_uint, cl_uint, cl_uint, cl_ulong, cl::Buffer, cl::Buffer, cl::Buffer>;
    using KernelTrainingApplyGradient = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl_uint, cl_ulong, cl_float, cl_float, cl_float>;

    mutable std::unique_ptr<KernelEval> m_kernel_calc_single_layer;
    mutable std::unique_ptr<KernelTrainingForwardPass> m_kernel_train_forward_pass;
    mutable std::unique_ptr<KernelTrainingBackwardPass> m_kernel_train_backward_pass;
    mutable std::unique_ptr<KernelTrainingApplyGradient> m_kernel_train_apply_gradient;

    cl::size_type m_kernel_calc_single_layer_ideal_workgroup_size = 64;
    cl::size_type m_kernel_training_ideal_workgroup_size = 16;
    cl::size_type m_kernel_training_apply_gradient_ideal_workgroup_size = 64;
    bool m_is_float16_supported = false;

  public:
    OpenCLComputeDevice(cl::Device device, OpenCLDeviceConfig advanced_config = {});

    virtual std::unique_ptr<NetworkResourceHandle> RegisterNetwork(Network& network) override;

    virtual std::vector<float> Evaluate(const NetworkResourceHandle& network_handle, std::span<const float> input) const override;

    virtual std::vector<float> EvaluateBatch(uint32_t batch_count, const NetworkResourceHandle& network_handle, std::span<const float> input) const override;

    virtual void ApplyRandomMutation(NetworkResourceHandle& network_handle, MutationDistribution weight_mutation_distribution, MutationDistribution bias_mutation_distribution) override;

    virtual void Train(NetworkResourceHandle& network, const TrainingSuite& training_suite, uint32_t trainingDataBegin, uint32_t trainingDataEnd) const override;

    static std::vector<cl::Device> GetDeviceList();

    static cl::Device AutoSelectDevice();

    std::string GetDeviceName() const override;

    size_t GetTotalMemory() const override;

    uint32_t GetComputeUnits() const override;

    bool SupportsWeightFormat(NetworkWeightFormat format) const override;
};
} // namespace macademy