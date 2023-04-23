#pragma once

#include "i_compute_backend.h"

namespace macademy
{
    class CPUComputeDevice : public IComputeDevice
    {
        public: 
        virtual NetworkResourceHandle RegisterNetwork(Network& network) override;

        virtual std::vector<float> Evaluate(const NetworkResourceHandle& network_handle, const std::vector<float>& input) const override;
    };
}