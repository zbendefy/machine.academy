#pragma once
#include <vector>
#include <span>
#include <memory>
#include <string>

namespace macademy
{
    class Network;

    struct NetworkResourceHandle
    {
        virtual ~NetworkResourceHandle(){}
        
        NetworkResourceHandle(Network& network) : m_network(&network) {}

        Network* m_network = nullptr;
    };

    class IComputeDevice
    {
        public:
        virtual ~IComputeDevice(){}

        virtual std::unique_ptr<NetworkResourceHandle> RegisterNetwork(Network& network) = 0;

        virtual std::vector<float> Evaluate(const NetworkResourceHandle& network, const std::span<float>& input) const = 0;

        virtual const std::string& GetDeviceName() const = 0;
    };
}