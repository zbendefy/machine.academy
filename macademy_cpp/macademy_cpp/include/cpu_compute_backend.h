#pragma once

#include "i_compute_backend.h"
#include <optional>

namespace macademy {
class CPUComputeDevice : public IComputeDevice
{
    const std::string m_name = "CPU device";

    void TrainOnMinibatch(const NetworkResourceHandle& network_handle, const TrainingSuite& training_suite);

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

    std::vector<float> EvaluateAndCollectInterimData(const NetworkResourceHandle& network_handle, const std::span<float>& input, std::optional<InterimTrainingData>& z_values) const;

  public:
    std::unique_ptr<NetworkResourceHandle> RegisterNetwork(Network& network) override;

    void Train(const NetworkResourceHandle& network, const TrainingSuite& training_suite) const override;

    std::vector<float> Evaluate(const NetworkResourceHandle& network_handle, const std::span<float>& input) const override;

    std::string GetDeviceName() const override;

    size_t GetTotalMemory() const override;

    uint32_t GetComputeUnits() const override;
};
} // namespace macademy