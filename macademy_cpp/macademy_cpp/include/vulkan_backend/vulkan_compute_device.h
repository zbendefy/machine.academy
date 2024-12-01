#pragma once

#include "i_compute_backend.h"
#include "vulkan_common.h"
#include "vulkan_backend/vulkan_device.h"
#include "vulkan_backend/vulkan_instance.h"
#include "vulkan_backend/vulkan_compute_pipeline.h"

#include <optional>

namespace macademy {

class VulkanComputeDevice : public IComputeDevice
{
    vk::Instance* m_instance;
    vk::Device* m_device;

    mutable std::unique_ptr<vk::ComputePipeline> m_kernel_calc_single_layer;

    //mutable std::unique_ptr<KernelEval> m_kernel_calc_single_layer;
    //mutable std::unique_ptr<KernelTrainingForwardPass> m_kernel_train_forward_pass;
    //mutable std::unique_ptr<KernelTrainingBackwardPass> m_kernel_train_backward_pass;
    //mutable std::unique_ptr<KernelTrainingApplyGradient> m_kernel_train_apply_gradient;

    uint32_t m_kernel_calc_single_layer_ideal_workgroup_size = 64;
    uint32_t m_kernel_training_ideal_workgroup_size = 16;
    uint32_t m_kernel_training_apply_gradient_ideal_workgroup_size = 64;
    bool m_is_float16_supported = false;

  public:
    VulkanComputeDevice(vk::Device* device);

    virtual std::unique_ptr<NetworkResourceHandle> RegisterNetwork(Network& network) override;

    virtual std::vector<float> Evaluate(const NetworkResourceHandle& network_handle, std::span<const float> input) const override;

    virtual std::vector<float> EvaluateBatch(uint32_t batch_count, const NetworkResourceHandle& network_handle, std::span<const float> input) const override;

    virtual void ApplyRandomMutation(NetworkResourceHandle& network_handle, MutationDistribution weight_mutation_distribution, MutationDistribution bias_mutation_distribution) override;

    virtual void Train(NetworkResourceHandle& network, const TrainingSuite& training_suite, uint32_t trainingDataBegin, uint32_t trainingDataEnd) const override;

    static std::vector<vk::Device*> GetDeviceList();

    static vk::Device* AutoSelectDevice();

    std::string GetDeviceName() const override;

    size_t GetTotalMemory() const override;

    uint32_t GetComputeUnits() const override;

    bool SupportsWeightFormat(NetworkWeightFormat format) const override;
};
} // namespace macademy::vk