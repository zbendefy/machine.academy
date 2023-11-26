#pragma once

#include "i_compute_backend.h"
#include "common.h"
#include <optional>

namespace macademy {
class CPUComputeDevice : public IComputeDevice
{
    const std::string m_name = "CPU device";

    std::vector<float> CalculateAccumulatedGradientForBatch(const NetworkResourceHandle& network_handle, const TrainingSuite& training_suite, uint32_t batch_begin, uint32_t batch_end) const;

    struct InterimTrainingData
    {
        std::vector<float> m_z_values;
        std::vector<float> m_activations;

        explicit InterimTrainingData(uint32_t data_size)
        {
            m_z_values.resize(data_size);
            m_activations.resize(data_size);
        }
    };

    void CalculateOutputLayerGradient(const Network& network, CostFunction cost_function, std::span<float> gradient_data, std::span<float> delta_k_vector, const InterimTrainingData& interim_data,
                                      const std::vector<float>& training_input, const std::vector<float>& desired_output) const;

    void CalculateHiddenLayerGradient(const Network& network, uint32_t layer_id, std::span<float> gradient_data, std::span<float> delta_k_vector, const InterimTrainingData& interim_data,
                                      const std::vector<float>& training_input) const;

    void EvaluateAndCollectInterimData(std::span<float> result_buffer, const NetworkResourceHandle& network_handle, std::span<const float> input, std::optional<InterimTrainingData>& z_values) const;

  public:
    std::unique_ptr<NetworkResourceHandle> RegisterNetwork(Network& network) override;

    void Train(NetworkResourceHandle& network, const TrainingSuite& training_suite, uint32_t trainingDataBegin, uint32_t trainingDataEnd) const override;

    std::vector<float> Evaluate(const NetworkResourceHandle& network_handle, std::span<const float> input) const override;

    std::vector<float> EvaluateBatch(uint32_t batch_size, const NetworkResourceHandle& network_handle, std::span<const float> input) const override;

    std::string GetDeviceName() const override;

    size_t GetTotalMemory() const override;

    uint32_t GetComputeUnits() const override;
};
} // namespace macademy