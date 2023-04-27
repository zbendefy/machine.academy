#pragma once

#include "i_compute_backend.h"

namespace macademy
{
    class CPUComputeDevice : public IComputeDevice
    {
        const std::string m_name = "CPU device";

        public: 
        std::unique_ptr<NetworkResourceHandle> RegisterNetwork(Network& network) override;

        std::vector<float> Evaluate(const NetworkResourceHandle& network_handle, const std::span<float>& input) const override;

        std::string GetDeviceName() const override;

        size_t GetTotalMemory() const override;
        
        uint32_t GetComputeUnits() const override;
    };
}