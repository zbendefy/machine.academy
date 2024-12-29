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

#include "opencl_kernels.cl"

void SetKernelArgs(cl::Kernel kernel, cl_uint index, OpenCLBuffer& buffer) { kernel.setArg(index, buffer.GetSize(), buffer.GetBuffer().get()); }

OpenCLComputeDevice::OpenCLComputeDevice(const ComputeDeviceInfo& device_info, OpenCLDeviceConfig advanced_config)
    : m_device(GetDeviceList()[device_info.m_device_index]), m_device_config(advanced_config), m_context(m_device), m_command_queue(m_context, m_device)
{
    auto extensions = m_device.getInfo<CL_DEVICE_EXTENSIONS>() + " ";

    m_is_float16_supported = extensions.find("cl_khr_fp16 ") != std::string::npos;

    std::vector<std::string> programStrings{opencl_kernel_source};
    m_program = cl::Program(m_context, programStrings);

    std::string args = ""; //"-cl-std=CL1.1";

#if CHECKED
    args += " -Werror";
#endif

    if (m_device_config.m_fast_relaxed_math) {
        args += " -cl-fast-relaxed-math";
    }

    if (m_device_config.m_mad_enable) {
        args += " -cl-mad-enable";
    }

    if (m_device_config.m_no_signed_zeros) {
        args += " -cl-no-signed-zeros";
    }

    if (m_device_config.m_unsafe_math_optimizations) {
        args += " -cl-unsafe-math-optimizations";
    }

    m_program.build(args.c_str());

    m_kernel_calc_single_layer = std::make_unique<KernelEval>(KernelEval(m_program, "evaluateLayerBatched"));
    m_kernel_calc_single_layer_ideal_workgroup_size = m_kernel_calc_single_layer->getKernel().getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(m_device, nullptr);

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

void OpenCLComputeDevice::QueueEvaluateLayerBatched(const IBuffer* weights_buffer, const IBuffer* layer_config_buffer, const IBuffer* layer_input_buffer, IBuffer* layer_output_buffer,
                                                    uint32_t layer_id, uint64_t weights_layer_offset, uint32_t batch_count, uint32_t layer_neuron_count)
{
    const auto weights_buffer_cl = BufferCast<const OpenCLBuffer>(weights_buffer);
    const auto layer_config_buffer_cl = BufferCast<const OpenCLBuffer>(layer_config_buffer);
    const auto layer_input_buffer_cl = BufferCast<const OpenCLBuffer>(layer_input_buffer);
    auto layer_output_buffer_cl = BufferCast<OpenCLBuffer>(layer_output_buffer);

    (*m_kernel_calc_single_layer)(cl::EnqueueArgs(m_command_queue, cl::NDRange(ExtendGlobalWorkSize(layer_neuron_count, m_kernel_calc_single_layer_ideal_workgroup_size), batch_count),
                                                  cl::NDRange(m_kernel_calc_single_layer_ideal_workgroup_size, 1)),
                                  weights_buffer_cl->GetBuffer(), layer_config_buffer_cl->GetBuffer(), layer_input_buffer_cl->GetBuffer(), layer_output_buffer_cl->GetBuffer(), layer_id,
                                  weights_layer_offset);
}

void OpenCLComputeDevice::QueueTrainForwardPass(const IBuffer* weights_buffer, const IBuffer* layer_config_buffer, const IBuffer* m_activations_zvalues_buffer, const IBuffer* input_buffer,
                                                uint32_t output_num, uint32_t layer_id, uint64_t weights_layer_offset, uint32_t num_training_samples, uint32_t total_neuron_count)
{
    const auto weights_buffer_cl = BufferCast<const OpenCLBuffer>(weights_buffer);
    const auto layer_config_buffer_cl = BufferCast<const OpenCLBuffer>(layer_config_buffer);
    const auto activations_zvalues_buffer_cl = BufferCast<const OpenCLBuffer>(m_activations_zvalues_buffer);
    const auto input_buffer_cl = BufferCast<OpenCLBuffer>(input_buffer);

    (*m_kernel_train_forward_pass)(cl::EnqueueArgs(m_command_queue,
                                                   cl::NDRange(ExtendGlobalWorkSize(output_num, m_kernel_calc_single_layer_ideal_workgroup_size),
                                                               ExtendGlobalWorkSize(num_training_samples, m_kernel_calc_single_layer_ideal_workgroup_size)),
                                                   cl::NDRange(m_kernel_training_ideal_workgroup_size, m_kernel_training_ideal_workgroup_size)),
                                   weights_buffer_cl->GetBuffer(), layer_config_buffer_cl->GetBuffer(), activations_zvalues_buffer_cl->GetBuffer(), input_buffer_cl->GetBuffer(), layer_id,
                                   weights_layer_offset, num_training_samples, total_neuron_count);
}

#if 0
void OpenCLComputeDevice::Train(NetworkResourceHandle& network_handle, const TrainingSuite& training_suite, uint32_t trainingDataBegin, uint32_t trainingDataEnd) const
{
    const auto opencl_network = dynamic_cast<const OpenCLNetworkResourceHandle*>(&network_handle);

    if (!opencl_network) {
        throw std::runtime_error("Network was not created by an OpenCL compute device!");
    }

    Network& network = *network_handle.m_network;

    const uint32_t num_training_samples = trainingDataEnd - trainingDataBegin;
    auto layer_config = network.GetLayerConfig();
    const uint32_t total_neuron_count = network.GetNeuronCount();
    const auto largest_layer_neuron_count = CalculateLargestLayerNeuronCount(layer_config);

    ASSERTM(opencl_network->m_input_buffer, "No training resources were allocated!");

    m_command_queue.enqueueFillBuffer<float>(opencl_network->m_gradient_buffer->GetBuffer(), 0.0f, 0, opencl_network->m_gradient_buffer->GetSize());

    std::vector<float> training_input_buffer_data;
    {
        training_input_buffer_data.resize(num_training_samples * network.GetInputCount());
        auto data_ptr = training_input_buffer_data.data();
        for (auto i = trainingDataBegin; i < trainingDataEnd; ++i) {
            std::memcpy(data_ptr, training_suite.m_training_data[i].m_input.data(), training_suite.m_training_data[i].m_input.size() * sizeof(float));
            data_ptr += training_suite.m_training_data[i].m_input.size();
        }

        m_command_queue.enqueueWriteBuffer(opencl_network->m_input_buffer->GetBuffer(), false, 0, training_input_buffer_data.size() * sizeof(float), training_input_buffer_data.data());
    }

    std::vector<float> training_desired_output_buffer_data;
    {
        training_desired_output_buffer_data.resize(num_training_samples * network.GetOutputCount());
        auto data_ptr = training_desired_output_buffer_data.data();
        for (auto i = trainingDataBegin; i < trainingDataEnd; ++i) {
            std::memcpy(data_ptr, training_suite.m_training_data[i].m_desired_output.data(), training_suite.m_training_data[i].m_desired_output.size() * sizeof(float));
            data_ptr += training_suite.m_training_data[i].m_desired_output.size();
        }

        m_command_queue.enqueueWriteBuffer(opencl_network->m_desired_output_buffer->GetBuffer(), false, 0, training_desired_output_buffer_data.size() * sizeof(float),
                                           training_desired_output_buffer_data.data());
    }

    cl_ulong weights_layer_offset = 0;

    // Forward pass (calculating z values and activations for each neuron times for each training data in the network)
    for (uint32_t i = 0; i < layer_config.size(); ++i) {
        const uint32_t input_num = i == 0 ? network.GetInputCount() : layer_config[i - 1].m_num_neurons;
        const uint32_t output_num = layer_config[i].m_num_neurons;

        (*m_kernel_train_forward_pass)(cl::EnqueueArgs(m_command_queue,
                                                       cl::NDRange(ExtendGlobalWorkSize(output_num, m_kernel_calc_single_layer_ideal_workgroup_size),
                                                                   ExtendGlobalWorkSize(num_training_samples, m_kernel_calc_single_layer_ideal_workgroup_size)),
                                                       cl::NDRange(m_kernel_training_ideal_workgroup_size, m_kernel_training_ideal_workgroup_size)),
                                       opencl_network->m_weights->GetBuffer(), opencl_network->m_layer_config_buffer->GetBuffer(), opencl_network->m_activations_zvalues_buffer->GetBuffer(),
                                       opencl_network->m_input_buffer->GetBuffer(), i, weights_layer_offset, num_training_samples, total_neuron_count);

        const cl_ulong layer_weight_size_bytes = cl_ulong(input_num) * output_num + output_num;
        ASSERTM(weights_layer_offset + layer_weight_size_bytes > weights_layer_offset, "Layer weights offset overflow!");
        weights_layer_offset += layer_weight_size_bytes; // advance the offset in the weights buffer for the next layer
    }

    // Backwards pass (accumulated gradient calculation)
    for (int i = layer_config.size() - 1; i >= 0; --i) {
        const uint32_t input_num = i == 0 ? network.GetInputCount() : layer_config[i - 1].m_num_neurons;
        const uint32_t output_num = layer_config[i].m_num_neurons;

        const cl_ulong layer_weight_size_bytes = cl_ulong(input_num) * output_num + output_num;
        weights_layer_offset -= layer_weight_size_bytes; // advance the offset backwards in the weights buffer

        (*m_kernel_train_backward_pass)(cl::EnqueueArgs(m_command_queue,
                                                        cl::NDRange(ExtendGlobalWorkSize(output_num, m_kernel_calc_single_layer_ideal_workgroup_size),
                                                                    ExtendGlobalWorkSize(num_training_samples, m_kernel_calc_single_layer_ideal_workgroup_size)),
                                                        cl::NDRange(m_kernel_training_ideal_workgroup_size, m_kernel_training_ideal_workgroup_size)),
                                        opencl_network->m_weights->GetBuffer(), opencl_network->m_layer_config_buffer->GetBuffer(), opencl_network->m_activations_zvalues_buffer->GetBuffer(),
                                        opencl_network->m_input_buffer->GetBuffer(), i, layer_config.size(), num_training_samples, total_neuron_count, cl_uint(training_suite.m_cost_function),
                                        largest_layer_neuron_count, weights_layer_offset, opencl_network->m_delta_k_buffer->GetBuffer(), opencl_network->m_gradient_buffer->GetBuffer(),
                                        opencl_network->m_desired_output_buffer->GetBuffer());
    }

    ASSERT(weights_layer_offset == 0);

    // Calculate regularization terms based on the training configuration
    float regularizationTerm1 = 1.0f;
    float regularizationTerm2Base = 0.0f;
    if (training_suite.m_regularization == Regularization::L2) {
        regularizationTerm1 = 1.0f - training_suite.m_learning_rate * (training_suite.m_regularization_rate / (float)training_suite.m_training_data.size());
    } else if (training_suite.m_regularization == Regularization::L1) {
        regularizationTerm2Base = -((training_suite.m_learning_rate * (training_suite.m_regularization_rate / (float)training_suite.m_training_data.size())));
    }
    const bool applyRegularizationTerm2 = regularizationTerm2Base != 0.0f;

    const float normalized_learning_rate = training_suite.m_learning_rate * (float(trainingDataEnd - trainingDataBegin) / (float)training_suite.m_training_data.size());

    // Gradient apply pass
    for (uint32_t i = 0; i < layer_config.size(); ++i) {
        const uint32_t input_num = i == 0 ? network.GetInputCount() : layer_config[i - 1].m_num_neurons;
        const uint32_t output_num = layer_config[i].m_num_neurons;

        (*m_kernel_train_apply_gradient)(cl::EnqueueArgs(m_command_queue, cl::NDRange(ExtendGlobalWorkSize(output_num, m_kernel_training_apply_gradient_ideal_workgroup_size)),
                                                         cl::NDRange(m_kernel_training_apply_gradient_ideal_workgroup_size)),
                                         opencl_network->m_weights->GetBuffer(), opencl_network->m_gradient_buffer->GetBuffer(), opencl_network->m_layer_config_buffer->GetBuffer(), i,
                                         weights_layer_offset, regularizationTerm1, regularizationTerm2Base, normalized_learning_rate);

        const cl_ulong layer_weight_size_bytes = cl_ulong(input_num) * output_num + output_num;
        ASSERTM(weights_layer_offset + layer_weight_size_bytes > weights_layer_offset, "Layer weights offset overflow!");
        weights_layer_offset += layer_weight_size_bytes; // advance the offset in the weights buffer for the next layer
    }

    m_command_queue.finish();
}
#endif

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

#if 0
void OpenCLComputeDevice::ApplyRandomMutation(NetworkResourceHandle& network_handle, MutationDistribution weight_mutation_distribution, MutationDistribution bias_mutation_distribution)
{
    auto opencl_network = dynamic_cast<OpenCLNetworkResourceHandle*>(&network_handle);

    if (!opencl_network) {
        throw std::runtime_error("Network was not created by an OpenCL compute device!");
    }

    Network& network = *network_handle.m_network;

    opencl_network->AllocateMutationBuffer();

    std::vector<float> mutation_buffer_data;
    mutation_buffer_data.resize(network.GetTotalWeightAndBiasCount());
    
    uint32_t weights_per_neuron = network.GetInputCount();
    uint32_t layer_neuron_count = 0;

    std::random_device rd;
    std::mt19937 gen(rd());

    auto generate_mutator = [&](const MutationDistribution& mutation_distribution) {
        if (std::holds_alternative<UniformDistribution>(mutation_distribution)) {
            UniformDistribution uniform_distribution_desc = std::get<UniformDistribution>(mutation_distribution);
            std::uniform_real_distribution uniform_distribution(-uniform_distribution_desc.range, uniform_distribution_desc.range);

            return [uniform_distribution, &gen](float x) mutable { return x + uniform_distribution(gen); };
        }
        throw std::runtime_error("Invalid mutation distribution!");
    };

    std::function<float(float)> weight_mutator = generate_mutator(weight_mutation_distribution);
    std::function<float(float)> bias_mutator = generate_mutator(bias_mutation_distribution);

    float* data_ptr = network.GetRawWeightData().data();

    const auto& layer_config = network.GetLayerConfig();

    for (size_t i = 0; i < layer_config.size(); ++i) {
        layer_neuron_count = layer_config[0].m_num_neurons;

        for (uint32_t n = 0; n < layer_neuron_count; ++n) {
            for (uint32_t w = 0; w < weights_per_neuron; ++w) {
                mutation_buffer_data.emplace_back(weight_mutator(0));
            }
            mutation_buffer_data.emplace_back(bias_mutator(0));
        }

        weights_per_neuron = layer_neuron_count;
    }

    m_command_queue.enqueueWriteBuffer(opencl_network->m_mutation_buffer->GetBuffer(), false, 0, mutation_buffer_data.size() * sizeof(float), mutation_buffer_data.data());

    cl_ulong weights_layer_offset = 0;

    for (uint32_t i = 0; i < layer_config.size(); ++i) {
        const uint32_t input_num = i == 0 ? network.GetInputCount() : layer_config[i - 1].m_num_neurons;
        const uint32_t output_num = layer_config[i].m_num_neurons;

        (*m_kernel_train_apply_gradient)(cl::EnqueueArgs(m_command_queue, cl::NDRange(ExtendGlobalWorkSize(output_num, m_kernel_training_apply_gradient_ideal_workgroup_size)),
                                                         cl::NDRange(m_kernel_training_apply_gradient_ideal_workgroup_size)),
                                         opencl_network->m_weights->GetBuffer(), opencl_network->m_gradient_buffer->GetBuffer(), opencl_network->m_layer_config_buffer->GetBuffer(), i,
                                         weights_layer_offset, 1.0f, 0.0f, -1.0f /*note: regularization_term_1 and 2 and learning rate are set to passtrough the modification*/);

        const cl_ulong layer_weight_size_bytes = cl_ulong(input_num) * output_num + output_num;
        ASSERTM(weights_layer_offset + layer_weight_size_bytes > weights_layer_offset, "Layer weights offset overflow!");
        weights_layer_offset += layer_weight_size_bytes; // advance the offset in the weights buffer for the next layer
    }

    m_command_queue.finish();
}
#endif

bool OpenCLComputeDevice::SupportsWeightFormat(NetworkWeightFormat format) const
{
    switch (format) {
    case macademy::NetworkWeightFormat::Float16:
        return m_is_float16_supported;
    case macademy::NetworkWeightFormat::Float32:
        return true;
    }

    throw std::runtime_error("OpenCLComputeDevice::SupportsWeightFormat: Invalid NetworkWeightFormat!");
}

} // namespace macademy