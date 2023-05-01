#pragma once
#include <vector>
#include <span>
#include <memory>
#include <string>

namespace macademy {
class Network;
class TrainingSuite;

struct NetworkResourceHandle
{
    virtual ~NetworkResourceHandle() {}

    NetworkResourceHandle(Network& network) : m_network(&network) {}

    Network* m_network = nullptr;
};

class IComputeDevice
{
  public:
    virtual ~IComputeDevice() {}

    virtual std::unique_ptr<NetworkResourceHandle> RegisterNetwork(Network& network) = 0;

    virtual std::vector<float> Evaluate(const NetworkResourceHandle& network, std::span<const float> input) const = 0;

    virtual void Train(const NetworkResourceHandle& network, const TrainingSuite& training_suite, uint32_t trainingDataBegin, uint32_t trainingDataEnd) const = 0;

    virtual std::string GetDeviceName() const = 0;

    virtual size_t GetTotalMemory() const = 0;

    virtual uint32_t GetComputeUnits() const = 0;
};
} // namespace macademy