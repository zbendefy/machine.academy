#include "opencl_backend/opencl_compute_device.h"
#include "opencl_backend/opencl_buffer.h"
#include "network.h"
#include "common.h"

#include <fstream>
#include <sstream>


namespace macademy
{   
    #include "opencl_kernels.cl"
    constexpr const char* calcLayerKernel = "calcSingleLayer";
    constexpr const char* forwardPass = "trainingForwardPass";
    constexpr const char* backwardPassKernel = "trainingBackwardPass";

    void SetKernelArgs(cl::Kernel kernel, cl_uint index, OpenCLBuffer& buffer)
    {
        kernel.setArg(index, buffer.GetSize(), buffer.GetBuffer().get());
    }

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

        OpenCLNetworkResourceHandle(cl::Context& context, Network& network)
            : m_context(context)
            , NetworkResourceHandle(network)
            {
                const size_t largest_layer_size_bytes = network.GetWeightByteSize() * std::max_element(network.GetLayerConfig().begin(), network.GetLayerConfig().end(), [](const LayerConfig& a, const LayerConfig& b){return a.m_num_neurons < b.m_num_neurons;})->m_num_neurons;

                std::vector<cl_uint> layer_config_buffer;

                {
                    layer_config_buffer.emplace_back(network.GetInputCount());
                    layer_config_buffer.emplace_back(0u); //dummy value to make input layer 2 wide
                    for (const auto& layer : network.GetLayerConfig())
                    {
                        layer_config_buffer.emplace_back(layer.m_num_neurons);
                        layer_config_buffer.emplace_back(uint32_t(layer.m_activation));
                    }
                }

                m_weights = std::make_unique<OpenCLBuffer>(m_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, network.GetRawWeightData().size_bytes(), network.GetRawWeightData().data());
                m_layer_config_buffer = std::make_unique<OpenCLBuffer>(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, layer_config_buffer.size() * sizeof(cl_uint), layer_config_buffer.data());
                m_layer_result_buffer_a = std::make_unique<OpenCLBuffer>(m_context, CL_MEM_READ_WRITE, largest_layer_size_bytes, nullptr);
                m_layer_result_buffer_b = std::make_unique<OpenCLBuffer>(m_context, CL_MEM_READ_WRITE, largest_layer_size_bytes, nullptr);
            }
    };

    OpenCLComputeDevice::OpenCLComputeDevice(cl::Device device)
        : m_device(device)
        , m_context(device)
        , m_command_queue(m_context, m_device)
    {
        std::vector<std::string> programStrings{
            opencl_kernel_source
        };
        m_program = cl::Program(m_context, programStrings);
        m_program.build("");

        m_kernel_calc_single_layer = std::make_unique<cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl_uint, cl_ulong>>(
            cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl_uint, cl_ulong>(m_program, "calcSingleLayer"));

            
        m_kernel_calc_single_layer_ideal_workgroup_size = m_kernel_calc_single_layer->getKernel().getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(m_device, nullptr);
    }

    std::unique_ptr<NetworkResourceHandle> OpenCLComputeDevice::RegisterNetwork(Network& network)
    {
        return std::make_unique<OpenCLNetworkResourceHandle>(m_context, network);
    }

    std::vector<float> OpenCLComputeDevice::Evaluate(const NetworkResourceHandle& network_handle, const std::span<float>& input) const
    {
        const auto opencl_network = dynamic_cast<const OpenCLNetworkResourceHandle*>(&network_handle);

        if (!opencl_network)
        {
            throw std::runtime_error("Network was not created by an OpenCL compute device!");
        }

        Network& network = *network_handle.m_network;

        if (input.size() != network.GetInputCount())
        {
            throw std::runtime_error("Invalid input length!");
        }

        auto layer_config = network.GetLayerConfig();

        auto layer_results_input = opencl_network->m_layer_result_buffer_a.get();
        auto layer_results_output = opencl_network->m_layer_result_buffer_b.get();

        //Write input into buffer
        m_command_queue.enqueueWriteBuffer(opencl_network->m_layer_result_buffer_a->GetBuffer(), true, 0, input.size_bytes(), input.data());

        cl_ulong weights_layer_offset = 0;

        float* neuron_weight_data = network.GetRawWeightData().data();
        for (uint32_t i = 0; i < layer_config.size(); ++i)
        {
            const uint32_t input_num = i == 0 ? input.size() : layer_config[i-1].m_num_neurons;
            const uint32_t output_num = layer_config[i].m_num_neurons;

            (*m_kernel_calc_single_layer)( cl::EnqueueArgs(m_command_queue, cl::NDRange(ExtendGlobalWorkSize(output_num, m_kernel_calc_single_layer_ideal_workgroup_size)), cl::NDRange(m_kernel_calc_single_layer_ideal_workgroup_size)), opencl_network->m_weights->GetBuffer(), opencl_network->m_layer_config_buffer->GetBuffer(), layer_results_input->GetBuffer(), layer_results_output->GetBuffer(), i, weights_layer_offset);

            const cl_ulong layer_weight_size_bytes = cl_ulong(input_num) * output_num + output_num;
            ASSERTM(weights_layer_offset + layer_weight_size_bytes > weights_layer_offset, "Layer weights offset overflow!");
            weights_layer_offset += layer_weight_size_bytes; //advance the offset in the weights buffer for the next layer

            std::swap(layer_results_input, layer_results_output); //output of this layer is input of the next
        }

        std::vector<float> result;
        result.resize(network.GetOutputCount());

        m_command_queue.enqueueReadBuffer(layer_results_input->GetBuffer(), true, 0, network.GetOutputCount() * network.GetWeightByteSize(), result.data());

        return result;
    }

    void OpenCLComputeDevice::Train(const NetworkResourceHandle& network, const TrainingSuite& training_suite) const
    {

    }

    std::vector<cl::Device> OpenCLComputeDevice::GetDeviceList()
    {
        std::vector<cl::Device> all_devices;
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        for (const auto& platform : platforms)
        {
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

    std::string OpenCLComputeDevice::GetDeviceName() const
    {
        return "OpenCL Device: " + m_device.getInfo<CL_DEVICE_NAME>();
    }

    size_t OpenCLComputeDevice::GetTotalMemory() const
    {
        return size_t(m_device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>());
    }

    uint32_t OpenCLComputeDevice::GetComputeUnits() const
    {
        return size_t(m_device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>());
    }
}