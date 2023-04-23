#pragma once

#include "i_compute_backend.h"

#include <CL/opencl.hpp>

namespace macademy
{
    class OpenCLComputeDevice : public IComputeDevice
    {
        cl::Device m_device;

        public: 
            OpenCLComputeDevice(cl::Device device);

        virtual std::unique_ptr<NetworkResourceHandle> RegisterNetwork(Network& network) override;

        virtual std::vector<float> Evaluate(const NetworkResourceHandle& network_handle, const std::span<float>& input) const override;

        static std::vector<cl::Device> GetDeviceList();

        static cl::Device AutoSelectDevice();
    };
}