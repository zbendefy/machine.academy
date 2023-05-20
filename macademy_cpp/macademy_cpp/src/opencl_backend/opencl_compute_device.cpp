#include "opencl_backend/opencl_compute_device.h"
#include "opencl_backend/opencl_buffer.h"
#include "network.h"
#include "common.h"
#include "utils.h"
#include "training_suite.h"

#include <fstream>
#include <sstream>

namespace macademy {
#include "opencl_kernels.cl"
constexpr const char* calcLayerKernel = "calcSingleLayer";
constexpr const char* forwardPass = "trainingForwardPass";
constexpr const char* backwardPassKernel = "trainingBackwardPass";

void SetKernelArgs(cl::Kernel kernel, cl_uint index, OpenCLBuffer& buffer) { kernel.setArg(index, buffer.GetSize(), buffer.GetBuffer().get()); }

size_t ExtendGlobalWorkSize(size_t desiredGlobalSize, size_t localSize)
{
    return ((desiredGlobalSize % localSize) == 0) ? desiredGlobalSize : (desiredGlobalSize + (localSize - (desiredGlobalSize % localSize)));
}

struct OpenCLNetworkResourceHandle : public NetworkResourceHandle
{
    cl::Context& m_context;
    std::unique_ptr<OpenCLBuffer> m_weights;
    std::unique_ptr<OpenCLBuffer> m_layer_config_buffer;
    std::unique_ptr<OpenCLBuffer> m_layer_result_buffer_a;
    std::unique_ptr<OpenCLBuffer> m_layer_result_buffer_b;

    OpenCLNetworkResourceHandle(cl::Context& context, cl::CommandQueue& command_queue, Network& network) : m_context(context), NetworkResourceHandle(network)
    {
        const size_t largest_layer_size_bytes = CalculateLargestLayerNeuronCount(network.GetLayerConfig()) * network.GetWeightByteSize();

        std::vector<cl_uint> layer_config_buffer;

        {
            layer_config_buffer.emplace_back(network.GetInputCount());
            layer_config_buffer.emplace_back(0u); // dummy value to make input layer 2 wide
            for (const auto& layer : network.GetLayerConfig()) {
                layer_config_buffer.emplace_back(layer.m_num_neurons);
                layer_config_buffer.emplace_back(uint32_t(layer.m_activation));
            }
        }

        m_weights = std::make_unique<OpenCLBuffer>(m_context, CL_MEM_READ_WRITE, network.GetRawWeightData().size_bytes(), nullptr);
        m_layer_config_buffer = std::make_unique<OpenCLBuffer>(m_context, CL_MEM_READ_ONLY, layer_config_buffer.size() * sizeof(cl_uint), nullptr);
        m_layer_result_buffer_a = std::make_unique<OpenCLBuffer>(m_context, CL_MEM_READ_WRITE, largest_layer_size_bytes, nullptr);
        m_layer_result_buffer_b = std::make_unique<OpenCLBuffer>(m_context, CL_MEM_READ_WRITE, largest_layer_size_bytes, nullptr);

        command_queue.enqueueWriteBuffer(m_weights->GetBuffer(), false, 0, network.GetRawWeightData().size_bytes(), network.GetRawWeightData().data());
        command_queue.enqueueWriteBuffer(m_weights->GetBuffer(), false, 0, layer_config_buffer.size() * sizeof(cl_uint), layer_config_buffer.data());
        command_queue.finish();

        //Note: using COPY_HOST_PTR at buffer creation is not optimal, as it will copy it first to host memory, and then upload it at kernel runtime
        //Enqueueing a writebuffer separately makes sure there are no 2 copies, and that the buffer is uploaded when this function returns.
        //See: https://stackoverflow.com/questions/3832963/what-is-the-difference-between-creating-a-buffer-object-with-clcreatebuffer-cl
    }
};

OpenCLComputeDevice::OpenCLComputeDevice(cl::Device device, OpenCLDeviceConfig advanced_config)
    : m_device(device), m_device_config(advanced_config), m_context(device), m_command_queue(m_context, m_device)
{
    std::vector<std::string> programStrings{opencl_kernel_source};
    m_program = cl::Program(m_context, programStrings);

    std::string args = "";//"-cl-std=CL1.1";

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

    m_kernel_calc_single_layer = std::make_unique<cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl_uint, cl_ulong>>(
        cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl_uint, cl_ulong>(m_program, "calcSingleLayer"));

    m_kernel_calc_single_layer_ideal_workgroup_size = m_kernel_calc_single_layer->getKernel().getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(m_device, nullptr);

    m_kernel_train_forward_pass = std::make_unique<cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl_uint, cl_ulong, cl_uint, cl_uint>>(
        cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl_uint, cl_ulong, cl_uint, cl_uint>(m_program, "trainingForwardPass"));
}

std::unique_ptr<NetworkResourceHandle> OpenCLComputeDevice::RegisterNetwork(Network& network) { return std::make_unique<OpenCLNetworkResourceHandle>(m_context, m_command_queue, network); }

std::vector<float> OpenCLComputeDevice::Evaluate(const NetworkResourceHandle& network_handle, std::span<const float> input) const
{
    const auto opencl_network = dynamic_cast<const OpenCLNetworkResourceHandle*>(&network_handle);

    if (!opencl_network) {
        throw std::runtime_error("Network was not created by an OpenCL compute device!");
    }

    Network& network = *network_handle.m_network;

    if (input.size() != network.GetInputCount()) {
        throw std::runtime_error("Invalid input length!");
    }

    auto layer_config = network.GetLayerConfig();

    auto layer_results_input = opencl_network->m_layer_result_buffer_a.get();
    auto layer_results_output = opencl_network->m_layer_result_buffer_b.get();

    // Write input into buffer
    m_command_queue.enqueueWriteBuffer(opencl_network->m_layer_result_buffer_a->GetBuffer(), false, 0, input.size_bytes(), input.data());

    cl_ulong weights_layer_offset = 0;

    float* neuron_weight_data = network.GetRawWeightData().data();
    for (uint32_t i = 0; i < layer_config.size(); ++i) {
        const uint32_t input_num = i == 0 ? input.size() : layer_config[i - 1].m_num_neurons;
        const uint32_t output_num = layer_config[i].m_num_neurons;

        (*m_kernel_calc_single_layer)(cl::EnqueueArgs(m_command_queue, cl::NDRange(ExtendGlobalWorkSize(output_num, m_kernel_calc_single_layer_ideal_workgroup_size)),
                                                      cl::NDRange(m_kernel_calc_single_layer_ideal_workgroup_size)),
                                      opencl_network->m_weights->GetBuffer(), opencl_network->m_layer_config_buffer->GetBuffer(), layer_results_input->GetBuffer(), layer_results_output->GetBuffer(),
                                      i, weights_layer_offset);

        const cl_ulong layer_weight_size_bytes = cl_ulong(input_num) * output_num + output_num;
        ASSERTM(weights_layer_offset + layer_weight_size_bytes > weights_layer_offset, "Layer weights offset overflow!");
        weights_layer_offset += layer_weight_size_bytes; // advance the offset in the weights buffer for the next layer

        std::swap(layer_results_input, layer_results_output); // output of this layer is input of the next
    }

    std::vector<float> result;
    result.resize(network.GetOutputCount());

    m_command_queue.enqueueReadBuffer(layer_results_input->GetBuffer(), true, 0, network.GetOutputCount() * network.GetWeightByteSize(), result.data());

    return result;
}

void OpenCLComputeDevice::Train(NetworkResourceHandle& network_handle, const TrainingSuite& training_suite, uint32_t trainingDataBegin, uint32_t trainingDataEnd) const
{
    const auto opencl_network = dynamic_cast<const OpenCLNetworkResourceHandle*>(&network_handle);

    if (!opencl_network) {
        throw std::runtime_error("Network was not created by an OpenCL compute device!");
    }

    Network& network = *network_handle.m_network;

    const uint32_t num_training_samples = trainingDataEnd - trainingDataBegin;
    const uint32_t total_neuron_count = network.GetNeuronCount();
    
    std::unique_ptr<OpenCLBuffer> m_input_buffer = std::make_unique<OpenCLBuffer>(m_context, CL_MEM_READ_ONLY, num_training_samples * network.GetInputCount() * sizeof(float), nullptr);
    std::unique_ptr<OpenCLBuffer> m_activations_zvalues_buffer = std::make_unique<OpenCLBuffer>(m_context, CL_MEM_READ_WRITE, network.GetNeuronCount() * 2 * sizeof(float), nullptr);

    
    std::vector<float> training_input_buffer_data;
    training_input_buffer_data.resize(num_training_samples * network.GetInputCount());
    auto input_ptr = training_input_buffer_data.data();
    for (auto i = trainingDataBegin; i < trainingDataEnd; ++i) {
        std::memcpy(input_ptr, training_suite.m_training_data[i].m_input.data(), training_suite.m_training_data[i].m_input.size() * sizeof(float));
        input_ptr += training_suite.m_training_data[i].m_input.size();
    }

    m_command_queue.enqueueWriteBuffer(m_input_buffer->GetBuffer(), false, 0, training_input_buffer_data.size() * sizeof(float), training_input_buffer_data.data());

    auto layer_config = network.GetLayerConfig();

    cl_ulong weights_layer_offset = 0;
    
    for (uint32_t i = 0; i < layer_config.size(); ++i) {
        const uint32_t input_num = i == 0 ? network.GetInputCount() : layer_config[i - 1].m_num_neurons;
        const uint32_t output_num = layer_config[i].m_num_neurons;

        (*m_kernel_train_forward_pass)(cl::EnqueueArgs(m_command_queue,
                                                       cl::NDRange(ExtendGlobalWorkSize(output_num, m_kernel_calc_single_layer_ideal_workgroup_size),
                                                                   ExtendGlobalWorkSize(num_training_samples, m_kernel_calc_single_layer_ideal_workgroup_size)),
                                                       cl::NDRange(m_kernel_training_ideal_workgroup_size, m_kernel_training_ideal_workgroup_size)),
                                      opencl_network->m_weights->GetBuffer(), opencl_network->m_layer_config_buffer->GetBuffer(), m_activations_zvalues_buffer->GetBuffer(), m_input_buffer->GetBuffer(), i, weights_layer_offset, num_training_samples, total_neuron_count);

        const cl_ulong layer_weight_size_bytes = cl_ulong(input_num) * output_num + output_num;
        ASSERTM(weights_layer_offset + layer_weight_size_bytes > weights_layer_offset, "Layer weights offset overflow!");
        weights_layer_offset += layer_weight_size_bytes; // advance the offset in the weights buffer for the next layer
    }

    m_command_queue.finish();
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

cl::Device OpenCLComputeDevice::AutoSelectDevice()
{
    auto all_devices = GetDeviceList();
    return all_devices[0];
}

std::string OpenCLComputeDevice::GetDeviceName() const { return "OpenCL Device: " + m_device.getInfo<CL_DEVICE_NAME>(); }

size_t OpenCLComputeDevice::GetTotalMemory() const { return size_t(m_device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>()); }

uint32_t OpenCLComputeDevice::GetComputeUnits() const { return size_t(m_device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>()); }
} // namespace macademy