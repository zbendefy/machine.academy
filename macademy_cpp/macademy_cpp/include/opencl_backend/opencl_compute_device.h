#pragma once

#include "i_compute_device.h"
#include "opencl_common.h"

#include <optional>
#include <nlohmann/json.hpp>

namespace macademy {

class OpenCLComputeDevice : public IComputeDevice
{
    cl::Device m_device;
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
    cl::size_type m_kernel_training_ideal_workgroup_size_x = 8;
    cl::size_type m_kernel_training_ideal_workgroup_size_y = 8;
    cl::size_type m_kernel_training_apply_gradient_ideal_workgroup_size = 64;
    bool m_is_float16_supported = false;

  public:
    OpenCLComputeDevice(const ComputeDeviceInfo& device, const nlohmann::json& device_config);

    std::unique_ptr<IBuffer> CreateBuffer(size_t size, BufferUsage buffer_usage, const std::string& name) override;

    void QueueWriteToBuffer(IBuffer* dst_buffer, std::span<const uint8_t> src, size_t buffer_offset) override;
    void QueueReadFromBuffer(IBuffer* src_buffer, std::span<uint8_t> dst, size_t buffer_offset) override;
    void QueueFillBuffer(IBuffer* buffer, uint32_t data, size_t offset, size_t size) override;
    void SubmitQueue() override;
    void WaitQueueIdle() override;

    void QueueEvaluateLayerBatched(const IBuffer* weights_buffer, const IBuffer* layer_config_buffer, const IBuffer* layer_input_buffer, IBuffer* layer_output_buffer, uint32_t current_layer_id,
                                   uint64_t current_layer_weights_offset, uint32_t batch_count, uint32_t layer_neuron_count) override;
    void QueueTrainForwardPass(const IBuffer* weights_buffer, const IBuffer* layer_config_buffer, IBuffer* m_activations_zvalues_buffer, const IBuffer* input_buffer, uint32_t layer_neuron_count,
                               uint32_t current_layer_id, uint64_t current_layer_weights_offset, uint32_t num_training_samples, uint32_t total_neuron_count) override;
    void QueueTrainBackwardPass(const IBuffer* weights_buffer, const IBuffer* layer_config_buffer, const IBuffer* m_activations_zvalues_buffer, const IBuffer* input_buffer, IBuffer* delta_k_vector,
                                IBuffer* gradient, const IBuffer* desiredOutputs, uint32_t layer_neuron_count, uint32_t current_layer_id, uint32_t layer_count, uint32_t numTrainingSamples,
                                uint32_t totalActivationCount, CostFunction costFunction, uint32_t largest_layer_neuron_count, uint64_t layer_weights_offset) override;
    void QueueApplyGradients(IBuffer* weights_buffer, const IBuffer* gradient_buffer, const IBuffer* layer_config_buffer, uint32_t layer_neuron_count, uint32_t current_layer_id,
                             uint64_t current_layer_weights_offset, float regularization_term_1, float regularization_term_2, float normalized_learning_rate) override;

    static std::vector<cl::Device> GetDeviceList();

    std::string GetDeviceName() const override;

    size_t GetTotalMemory() const override;

    bool SupportsWeightFormat(NetworkWeightFormat format) const override;

    static std::vector<ComputeDeviceInfo> GetOpenCLComputeDeviceInfo();
};

} // namespace macademy