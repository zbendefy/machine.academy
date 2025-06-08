#include "opencl_backend/opencl_compute_device.h"
#include "opencl_backend/opencl_buffer.h"
#include "network.h"
#include "common.h"
#include "utils.h"
#include "training_suite.h"

#include <fstream>
#include <sstream>

namespace {
size_t ExtendGlobalWorkSize(size_t desiredGlobalSize, size_t localSize)
{
    return ((desiredGlobalSize % localSize) == 0) ? desiredGlobalSize : (desiredGlobalSize + (localSize - (desiredGlobalSize % localSize)));
}

cl_mem_flags ToOpenCLBufferUsage(macademy::BufferUsage usage)
{
    switch (usage) {
    case macademy::BufferUsage::ReadOnly:
        return CL_MEM_READ_ONLY;
    case macademy::BufferUsage::ReadWrite:
        return CL_MEM_READ_WRITE;
    case macademy::BufferUsage::WriteOnly:
        return CL_MEM_WRITE_ONLY;
    }

    throw std::runtime_error("invalid buffer usage!");
}

} // namespace

namespace macademy {

std::vector<ComputeDeviceInfo> OpenCLComputeDevice::GetOpenCLComputeDeviceInfo()
{
    auto devices = OpenCLComputeDevice::GetDeviceList();

    std::vector<ComputeDeviceInfo> ret;

    uint32_t idx = 0;

    for (auto& it : devices) {
        ret.push_back(
            ComputeDeviceInfo{.m_backend = "opencl", .m_device_index = idx++, .m_device_name = it.getInfo<CL_DEVICE_NAME>(), .m_total_memory = uint64_t(it.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>())});
    }

    return ret;
}

#include "opencl_kernels.cl.h"

void SetKernelArgs(cl::Kernel kernel, cl_uint index, OpenCLBuffer& buffer) { kernel.setArg(index, buffer.GetSize(), buffer.GetBuffer().get()); }

OpenCLComputeDevice::OpenCLComputeDevice(const ComputeDeviceInfo& device_info, const nlohmann::json& device_config)
    : m_device(GetDeviceList()[device_info.m_device_index]), m_context(m_device), m_command_queue(m_context, m_device)
{
    auto extensions = m_device.getInfo<CL_DEVICE_EXTENSIONS>() + " ";

    m_is_float16_supported = extensions.find("cl_khr_fp16 ") != std::string::npos;

    std::vector<std::string> programStrings{opencl_kernel_source};
    m_program = cl::Program(m_context, programStrings);

    std::string args = ""; //"-cl-std=CL1.1";

#if CHECKED
    args += " -Werror";
#endif

    if (GetBoolFlagFromJson(device_config, "cl_fast_relaxed_math", true)) {
        args += " -cl-fast-relaxed-math";
    }

    if (GetBoolFlagFromJson(device_config, "cl_mad_enable", true)) {
        args += " -cl-mad-enable";
    }

    if (GetBoolFlagFromJson(device_config, "cl_no_signed_zeros", true)) {
        args += " -cl-no-signed-zeros";
    }

    if (GetBoolFlagFromJson(device_config, "cl_unsafe_math_operations", false)) {
        args += " -cl-unsafe-math-optimizations";
    }

    m_program.build(args.c_str());

    m_kernel_calc_single_layer = std::make_unique<KernelEval>(KernelEval(m_program, "evaluateLayerBatched"));
    m_kernel_calc_single_layer_ideal_workgroup_size = m_kernel_calc_single_layer->getKernel().getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(m_device, nullptr);

    m_kernel_calc_single_layer_ideal_workgroup_size = GetIntFromJson(device_config, "eval_threadgroup_size", m_kernel_calc_single_layer_ideal_workgroup_size);
    m_kernel_training_ideal_workgroup_size_x = GetIntFromJson(device_config, "training_threadgroup_size_x", m_kernel_training_ideal_workgroup_size_x);
    m_kernel_training_ideal_workgroup_size_y = GetIntFromJson(device_config, "training_threadgroup_size_x", m_kernel_training_ideal_workgroup_size_y);
    m_kernel_training_apply_gradient_ideal_workgroup_size = GetIntFromJson(device_config, "gradient_apply_threadgroup_size", m_kernel_training_apply_gradient_ideal_workgroup_size);

    m_kernel_train_forward_pass = std::make_unique<KernelTrainingForwardPass>(KernelTrainingForwardPass(m_program, "trainingForwardPass"));
    m_kernel_train_backward_pass = std::make_unique<KernelTrainingBackwardPass>(KernelTrainingBackwardPass(m_program, "trainingBackwardPass"));
    m_kernel_train_apply_gradient = std::make_unique<KernelTrainingApplyGradient>(KernelTrainingApplyGradient(m_program, "trainingApplyGradient"));
}

std::unique_ptr<IBuffer> OpenCLComputeDevice::CreateBuffer(size_t size, BufferUsage buffer_usage, const std::string& name)
{
    auto ret = std::make_unique<OpenCLBuffer>(m_context, ToOpenCLBufferUsage(buffer_usage), size, nullptr);

    return ret;
}

void OpenCLComputeDevice::QueueWriteToBuffer(IBuffer* dst_buffer, std::span<const uint8_t> src, size_t buffer_offset)
{
    auto cl_buffer = BufferCast<OpenCLBuffer>(dst_buffer);

    m_command_queue.enqueueWriteBuffer(cl_buffer->GetBuffer(), false, cl::size_type(buffer_offset), cl::size_type(src.size()), src.data());
}

void OpenCLComputeDevice::QueueReadFromBuffer(IBuffer* src_buffer, std::span<uint8_t> dst, size_t buffer_offset)
{
    auto cl_buffer = BufferCast<OpenCLBuffer>(src_buffer);

    m_command_queue.enqueueReadBuffer(cl_buffer->GetBuffer(), false, cl::size_type(buffer_offset), cl::size_type(dst.size()), dst.data());
}

void OpenCLComputeDevice::QueueFillBuffer(IBuffer* buffer, uint32_t data, size_t offset_bytes, size_t size_bytes)
{
    auto cl_buffer = BufferCast<OpenCLBuffer>(buffer);

    m_command_queue.enqueueFillBuffer(cl_buffer->GetBuffer(), cl_uint(data), cl::size_type(offset_bytes), cl::size_type(size_bytes));
}

void OpenCLComputeDevice::SubmitQueue() { m_command_queue.flush(); }

void OpenCLComputeDevice::WaitQueueIdle() { m_command_queue.finish(); }

void OpenCLComputeDevice::QueueEvaluateLayer(const IBuffer* tensor_buffer, const IBuffer* layer_input_buffer, IBuffer* layer_output_buffer, ActivationFunction activation,
    uint32_t layer_input_count, uint32_t layer_neuron_count)
{
#if 0
    const auto weights_buffer_cl = BufferCast<const OpenCLBuffer>(weights_buffer);
    const auto layer_config_buffer_cl = BufferCast<const OpenCLBuffer>(layer_config_buffer);
    const auto layer_input_buffer_cl = BufferCast<const OpenCLBuffer>(layer_input_buffer);
    auto layer_output_buffer_cl = BufferCast<OpenCLBuffer>(layer_output_buffer);

    (*m_kernel_calc_single_layer)(cl::EnqueueArgs(m_command_queue, cl::NDRange(ExtendGlobalWorkSize(layer_neuron_count, m_kernel_calc_single_layer_ideal_workgroup_size), batch_count),
                                                  cl::NDRange(m_kernel_calc_single_layer_ideal_workgroup_size, 1)),
                                  weights_buffer_cl->GetBuffer(), layer_config_buffer_cl->GetBuffer(), layer_input_buffer_cl->GetBuffer(), layer_output_buffer_cl->GetBuffer(), current_layer_id,
                                  current_layer_weights_offset);
#endif
}

void OpenCLComputeDevice::QueueTrainForwardPass(const IBuffer* tensor_buffer, const IBuffer* prev_activations, IBuffer* activations, IBuffer* zvalues, ActivationFunction activation_function,
    uint32_t layer_neuron_count, uint32_t weights_per_neuron, uint32_t num_training_samples)
{
#if 0
    const auto weights_buffer_cl = BufferCast<const OpenCLBuffer>(weights_buffer);
    const auto layer_config_buffer_cl = BufferCast<const OpenCLBuffer>(layer_config_buffer);
    const auto activations_zvalues_buffer_cl = BufferCast<const OpenCLBuffer>(m_activations_zvalues_buffer);
    const auto input_buffer_cl = BufferCast<const OpenCLBuffer>(input_buffer);

    (*m_kernel_train_forward_pass)(cl::EnqueueArgs(m_command_queue,
                                                   cl::NDRange(ExtendGlobalWorkSize(layer_neuron_count, m_kernel_training_ideal_workgroup_size_x),
                                                               ExtendGlobalWorkSize(num_training_samples, m_kernel_training_ideal_workgroup_size_y)),
                                                   cl::NDRange(m_kernel_training_ideal_workgroup_size_x, m_kernel_training_ideal_workgroup_size_y)),
                                   weights_buffer_cl->GetBuffer(), layer_config_buffer_cl->GetBuffer(), activations_zvalues_buffer_cl->GetBuffer(), input_buffer_cl->GetBuffer(), current_layer_id,
                                   current_layer_weights_offset, num_training_samples, total_neuron_count);
#endif
}

void OpenCLComputeDevice::QueueTrainBackwardPass(const IBuffer* next_layer_tensor_buffer, const IBuffer* prev_activations_buffer, const IBuffer* layer_activations_buffer, const IBuffer* layer_zvalues_buffer,
    IBuffer* delta_k_vector_buffer_write, const IBuffer* delta_k_vector_buffer_read, IBuffer* current_layer_gradient_buffer, const IBuffer* desiredOutputsBuffer, uint32_t layer_neuron_count, uint32_t weights_per_neuron, ActivationFunction activation_function, uint32_t numTrainingSamples, CostFunction costFunction, uint32_t next_layer_neuron_count)
{
#if 0
    const auto weights_buffer_cl = BufferCast<const OpenCLBuffer>(weights_buffer);
    const auto layer_config_buffer_cl = BufferCast<const OpenCLBuffer>(layer_config_buffer);
    const auto activations_zvalues_buffer_cl = BufferCast<const OpenCLBuffer>(m_activations_zvalues_buffer);
    const auto input_buffer_cl = BufferCast<const OpenCLBuffer>(input_buffer);
    const auto delta_k_vector_cl = BufferCast<const OpenCLBuffer>(delta_k_vector);
    const auto gradient_cl = BufferCast<const OpenCLBuffer>(gradient);
    const auto desiredOutputs_cl = BufferCast<const OpenCLBuffer>(desiredOutputs);

    (*m_kernel_train_backward_pass)(cl::EnqueueArgs(m_command_queue,
                                                    cl::NDRange(ExtendGlobalWorkSize(layer_neuron_count, m_kernel_training_ideal_workgroup_size_x),
                                                                ExtendGlobalWorkSize(numTrainingSamples, m_kernel_training_ideal_workgroup_size_y)),
                                                    cl::NDRange(m_kernel_training_ideal_workgroup_size_x, m_kernel_training_ideal_workgroup_size_y)),
                                    weights_buffer_cl->GetBuffer(), layer_config_buffer_cl->GetBuffer(), activations_zvalues_buffer_cl->GetBuffer(), input_buffer_cl->GetBuffer(), current_layer_id,
                                    layer_count, numTrainingSamples, total_neuron_count, cl_uint(costFunction), largest_layer_neuron_count, layer_weights_offset, delta_k_vector_cl->GetBuffer(),
                                    gradient_cl->GetBuffer(), desiredOutputs_cl->GetBuffer());
#endif
}

void OpenCLComputeDevice::QueueApplyGradients(IBuffer* tensor_buffer, const IBuffer* gradient_buffer, uint32_t layer_neuron_count, uint32_t weights_per_neuron, float regularization_term_1, float regularization_term_2, float normalized_learning_rate)
{

#if 0
    const auto weights_buffer_cl = BufferCast<const OpenCLBuffer>(weights_buffer);
    const auto layer_config_buffer_cl = BufferCast<const OpenCLBuffer>(layer_config_buffer);
    const auto gradient_cl = BufferCast<const OpenCLBuffer>(gradient_buffer);

    (*m_kernel_train_apply_gradient)(cl::EnqueueArgs(m_command_queue, cl::NDRange(ExtendGlobalWorkSize(layer_neuron_count, m_kernel_training_apply_gradient_ideal_workgroup_size)),
                                                     cl::NDRange(m_kernel_training_apply_gradient_ideal_workgroup_size)),
                                     weights_buffer_cl->GetBuffer(), gradient_cl->GetBuffer(), layer_config_buffer_cl->GetBuffer(), current_layer_id, current_layer_weights_offset,
                                     regularization_term_1, regularization_term_2, normalized_learning_rate);
#endif
}

std::vector<cl::Device> OpenCLComputeDevice::GetDeviceList()
{
    std::vector<cl::Device> all_devices;
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    for (const auto& platform : platforms) {
        std::vector<cl::Device> platform_devices;
        platform.getDevices(CL_DEVICE_TYPE_ALL, &platform_devices);

        std::copy(platform_devices.begin(), platform_devices.end(), std::back_inserter(all_devices));
    }

    return all_devices;
}

std::string OpenCLComputeDevice::GetDeviceName() const { return "OpenCL Device: " + m_device.getInfo<CL_DEVICE_NAME>(); }

size_t OpenCLComputeDevice::GetTotalMemory() const { return size_t(m_device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>()); }

bool OpenCLComputeDevice::SupportsWeightFormat(DType format) const
{
    switch (format) {
    case macademy::DType::Float16:
        return m_is_float16_supported;
    case macademy::DType::Float32:
        return true;
    }

    throw std::runtime_error("OpenCLComputeDevice::SupportsWeightFormat: Invalid DType!");
}

} // namespace macademy