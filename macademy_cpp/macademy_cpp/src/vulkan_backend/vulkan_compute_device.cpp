#include "vulkan_backend/vulkan_compute_device.h"
#include "vulkan_backend/vulkan_buffer.h"
#include "vulkan_backend/shader_specialization.h"
#include "network.h"
#include "common.h"
#include "utils.h"
#include "training_suite.h"

#include <fstream>
#include <sstream>

namespace macademy {
//#include "opencl_kernels.cl"

class VulkanNetworkResourceHandle : public NetworkResourceHandle
{
    vk::Device* m_device{};
    std::unique_ptr<vk::VulkanBuffer> m_weights_buffer;
    std::unique_ptr<vk::VulkanBuffer> m_layer_config_buffer;
    mutable std::unique_ptr<vk::VulkanBuffer> m_layer_result_buffer_a;
    mutable std::unique_ptr<vk::VulkanBuffer> m_layer_result_buffer_b;

  public:
    VulkanNetworkResourceHandle(vk::Device* device, Network& network) : m_device(device), NetworkResourceHandle(network)
    {
        std::vector<uint32_t> layer_config_buffer;

        {
            layer_config_buffer.emplace_back(network.GetInputCount());
            layer_config_buffer.emplace_back(0u); // dummy value to make input layer 2 wide
            for (const auto& layer : network.GetLayerConfig()) {
                layer_config_buffer.emplace_back(layer.m_num_neurons);
                layer_config_buffer.emplace_back(uint32_t(layer.m_activation));
            }
        }

        m_weights_buffer = std::make_unique<vk::VulkanBuffer>(m_device, "m_weights_buffer", network.GetRawWeightData().size_bytes(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                              VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);

        //This is small enough to fit into the faster but smaller (at least 16kb) uniform buffer storage. (Note: on AMD/Intel gpus its not faster than storage buffers though)
        m_layer_config_buffer = std::make_unique<vk::VulkanBuffer>(m_device, "m_layer_config_buffer", layer_config_buffer.size() * sizeof(uint32_t), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                                                   VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);
        
        //This upload uses 2 commandbuffers and submits, TODO: upload these buffers in a single command buffer and submit.
        m_weights_buffer->UpdateData(std::span<uint8_t>(std::bit_cast<uint8_t*>(network.GetRawWeightData().data()), network.GetRawWeightData().size_bytes()), 0);
        m_layer_config_buffer->UpdateData(std::span<uint8_t>(std::bit_cast<uint8_t*>(network.GetRawWeightData().data()), network.GetRawWeightData().size() * sizeof(uint32_t)), 0);
    }

    void SynchronizeNetworkData() override {}

    void AllocateBatchEvalResources(uint32_t batch_count) const
    {
        const size_t largest_layer_size_bytes = std::max(m_network->GetInputCount(), CalculateLargestLayerNeuronCount(m_network->GetLayerConfig())) * m_network->GetWeightByteSize();
        const size_t largest_layer_buffer_required_size = largest_layer_size_bytes * batch_count;
        
        if (!m_layer_result_buffer_a || m_layer_result_buffer_a->GetSize() < largest_layer_buffer_required_size) {
            m_layer_result_buffer_a.reset();
            m_layer_result_buffer_a = std::make_unique<vk::VulkanBuffer>(m_device, "m_layer_result_buffer_a", largest_layer_buffer_required_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                                         VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);
        }

        if (!m_layer_result_buffer_b || m_layer_result_buffer_b->GetSize() < largest_layer_buffer_required_size) {
            m_layer_result_buffer_b.reset();
            m_layer_result_buffer_b = std::make_unique<vk::VulkanBuffer>(m_device, "m_layer_result_buffer_b", largest_layer_buffer_required_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                                         VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);
        }
    }

    void AllocateMutationBuffer()
    {
    }

    virtual void AllocateTrainingResources(uint32_t training_sample_count)
    {
    }

    virtual void FreeCachedResources()
    {
        //m_input_buffer.reset();
        //m_desired_output_buffer.reset();
        //m_activations_zvalues_buffer.reset();
        //m_delta_k_buffer.reset();
        //m_gradient_buffer.reset();
        m_layer_result_buffer_a.reset();
        m_layer_result_buffer_b.reset();
        //m_mutation_buffer.reset();
    }
};

VulkanComputeDevice::VulkanComputeDevice(vk::Device* device) : m_device(device)
{
    vk::ShaderSpecializationMap shader_specialization;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &uboLayoutBinding;

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor set layout!");
    }
    vk::ComputePipelineDescriptor kernel_calc_single_layer_pipeline_desc{};
    kernel_calc_single_layer_pipeline_desc.m_compute_shader = 0;
    kernel_calc_single_layer_pipeline_desc.m_compute_shader_entry_function = "kernel_calc_single_layer";
    kernel_calc_single_layer_pipeline_desc.m_descriptor_set_layout = ;
    kernel_calc_single_layer_pipeline_desc.m_push_constant_size = 0;

    m_kernel_calc_single_layer = std::make_unique<vk::ComputePipeline>(m_device, "kernel_calc_single_layer", kernel_calc_single_layer_pipeline_desc, shader_specialization);
    
}

std::unique_ptr<NetworkResourceHandle> VulkanComputeDevice::RegisterNetwork(Network& network) { return std::make_unique<VulkanNetworkResourceHandle>(network); }

std::vector<float> VulkanComputeDevice::Evaluate(const NetworkResourceHandle& network_handle, std::span<const float> input) const { return EvaluateBatch(1, network_handle, input); }

std::vector<float> VulkanComputeDevice::EvaluateBatch(uint32_t batch_count, const NetworkResourceHandle& network_handle, std::span<const float> input) const 
{ return {}; }

void VulkanComputeDevice::Train(NetworkResourceHandle& network_handle, const TrainingSuite& training_suite, uint32_t trainingDataBegin, uint32_t trainingDataEnd) const
{
}

std::vector<vk::Device*> VulkanComputeDevice::GetDeviceList()
{
    static vk::Instance instance{true, true};
    return instance.GetDevices();
}

std::string VulkanComputeDevice::GetDeviceName() const { return "Vulkan Device: " + m_device->GetName(); }

size_t VulkanComputeDevice::GetTotalMemory() const { return 0; }

uint32_t VulkanComputeDevice::GetComputeUnits() const { return 0; }

void VulkanComputeDevice::ApplyRandomMutation(NetworkResourceHandle& network_handle, MutationDistribution weight_mutation_distribution, MutationDistribution bias_mutation_distribution)
{
}

bool VulkanComputeDevice::SupportsWeightFormat(NetworkWeightFormat format) const
{
    switch (format) {
    case macademy::NetworkWeightFormat::Float16:
        return m_is_float16_supported;
    case macademy::NetworkWeightFormat::Float32:
        return true;
    }

    throw std::runtime_error("VulkanComputeDevice::SupportsWeightFormat: Invalid NetworkWeightFormat!");
}

} // namespace macademy