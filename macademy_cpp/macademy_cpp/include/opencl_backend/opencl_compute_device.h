#pragma once

#include "i_compute_device.h"
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
    OpenCLComputeDevice(const ComputeDeviceInfo& device, OpenCLDeviceConfig advanced_config = {});

    std::unique_ptr<IBuffer> CreateBuffer(size_t size, BufferUsage buffer_usage, const std::string& name) override;

    void QueueWriteToBuffer(IBuffer* dst_buffer, std::span<const uint8_t> src, size_t buffer_offset) override;
    void QueueReadFromBuffer(IBuffer* src_buffer, std::span<uint8_t> dst, size_t buffer_offset) override;
    void QueueFillBuffer(IBuffer* buffer, uint32_t data, size_t offset, size_t size) override;
    void SubmitQueue() override;
    void WaitQueueIdle() override;

    void QueueEvaluateLayerBatched(const IBuffer* weights_buffer, const IBuffer* layer_config_buffer, const IBuffer* layer_input_buffer, IBuffer* layer_output_buffer, uint32_t layer_id,
                                   uint64_t weights_layer_offset, uint32_t batch_count, uint32_t layer_neuron_count) override;

    static std::vector<cl::Device> GetDeviceList();

    std::string GetDeviceName() const override;

    size_t GetTotalMemory() const override;

    bool SupportsWeightFormat(NetworkWeightFormat format) const override;

    static std::vector<ComputeDeviceInfo> GetOpenCLComputeDeviceInfo();
};

} // namespace macademy