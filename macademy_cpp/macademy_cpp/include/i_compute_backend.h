#pragma once
#include <vector>
#include <span>
#include <memory>
#include <string>
#include <variant>

namespace macademy {
class Network;
class TrainingSuite;

struct UniformDistribution 
{
    float range;
};

enum class NetworkWeightFormat
{
    Float16, Float32
};

/// <summary>
/// A class representing an opaque handle to a neural network compiled for a specific device
/// </summary>
struct NetworkResourceHandle
{
    virtual ~NetworkResourceHandle() {}

    NetworkResourceHandle(Network& network) : m_network(&network) {}

    virtual void SynchronizeNetworkData() {}

    virtual void AllocateTrainingResources(uint32_t training_sample_count) {}

    virtual void FreeCachedResources() {}

    Network* m_network = nullptr;
};

using MutationDistribution = std::variant<UniformDistribution>;

class IComputeDevice
{
  public:
    virtual ~IComputeDevice() {}

    virtual std::unique_ptr<NetworkResourceHandle> RegisterNetwork(Network& network) = 0;

    virtual std::vector<float> Evaluate(const NetworkResourceHandle& network, std::span<const float> input) const = 0;

    virtual std::vector<float> EvaluateBatch(uint32_t batch_size, const NetworkResourceHandle& network, std::span<const float> input) const = 0;

    virtual void ApplyRandomMutation(NetworkResourceHandle& network_handle, MutationDistribution weight_mutation_distribution, MutationDistribution bias_mutation_distribution) = 0;

    virtual void Train(NetworkResourceHandle& network, const TrainingSuite& training_suite, uint32_t trainingDataBegin, uint32_t trainingDataEnd) const = 0;

    virtual std::string GetDeviceName() const = 0;

    virtual size_t GetTotalMemory() const = 0;

    virtual bool SupportsWeightFormat(NetworkWeightFormat format) const = 0;

    virtual uint32_t GetComputeUnits() const = 0;
};
} // namespace macademy