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

    using KernelEval = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl_uint, cl_uint, cl_uint>;
    using KernelTrainingForwardPass = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl_uint, cl_uint, cl_uint, cl_uint>;
    using KernelTrainingBackwardPass =
        cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl_uint, cl_uint, cl_uint, cl_uint, cl_uint, cl_uint, cl_uint>;
    using KernelTrainingApplyGradient = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl_uint, cl_uint, cl_float, cl_float, cl_float>;

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

    void QueueEvaluateLayer(const IBuffer* tensor_buffer, const IBuffer* layer_input_buffer, IBuffer* layer_output_buffer, ActivationFunction activation,
        uint32_t layer_input_count, uint32_t layer_neuron_count) override;
    void QueueTrainForwardPass(const IBuffer* tensor_buffer, const IBuffer* prev_activations, IBuffer* activations, IBuffer* zvalues, ActivationFunction activation_function,
        uint32_t layer_neuron_count, uint32_t weights_per_neuron, uint32_t num_training_samples) override;
    void QueueTrainBackwardPass(const IBuffer* next_layer_tensor_buffer, const IBuffer* prev_activations_buffer, const IBuffer* layer_activations_buffer, const IBuffer* layer_zvalues_buffer,
        IBuffer* delta_k_vector_buffer_write, const IBuffer* delta_k_vector_buffer_read, IBuffer* current_layer_gradient_buffer, const IBuffer* desiredOutputsBuffer, uint32_t layer_neuron_count, uint32_t weights_per_neuron, ActivationFunction activation_function, uint32_t numTrainingSamples, CostFunction costFunction, uint32_t next_layer_neuron_count) override;
    void QueueApplyGradients(IBuffer* tensor_buffer, const IBuffer* gradient_buffer, uint32_t layer_neuron_count, uint32_t weights_per_neuron, float regularization_term_1, float regularization_term_2, float normalized_learning_rate) override;

    static std::vector<cl::Device> GetDeviceList();

    std::string GetDeviceName() const override;

    size_t GetTotalMemory() const override;

    bool SupportsWeightFormat(DType format) const override;

    static std::vector<ComputeDeviceInfo> GetOpenCLComputeDeviceInfo();
};

} // namespace macademy