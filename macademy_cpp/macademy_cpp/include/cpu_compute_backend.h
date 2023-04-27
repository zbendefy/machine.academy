#pragma once

#include "i_compute_backend.h"

namespace macademy
{
    class CPUComputeDevice : public IComputeDevice
    {
        const std::string m_name = "CPU device";

        public: 
        virtual std::unique_ptr<NetworkResourceHandle> RegisterNetwork(Network& network) override;

        virtual std::vector<float> Evaluate(const NetworkResourceHandle& network_handle, const std::span<float>& input) const override;

        const std::string& GetDeviceName() const override { return m_name; }
    };
}