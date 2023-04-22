#pragma once
#include <vector>

namespace macademy
{
    class Network;

    struct NetworkResourceHandle
    {
        Network* m_network = nullptr;
    };

    class IComputeDevice
    {
        public:
        virtual ~IComputeDevice(){}

        virtual NetworkResourceHandle RegisterNetwork(Network& network) = 0;

        virtual std::vector<float> Evaluate(const NetworkResourceHandle& network, const std::vector<float>& input) const = 0;
    };
}