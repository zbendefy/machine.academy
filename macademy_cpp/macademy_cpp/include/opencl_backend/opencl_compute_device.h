#pragma once

#include "i_compute_backend.h"

#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 100
#define CL_HPP_CL_1_2_DEFAULT_BUILD 1
#define CL_HPP_ENABLE_EXCEPTIONS 1
#include <CL/opencl.hpp>

namespace macademy
{
    class OpenCLComputeDevice : public IComputeDevice
    {
        cl::Device m_device;
        cl::CommandQueue m_command_queue;
        cl::Program m_program;
        std::unique_ptr<cl::KernelFunctor<>> m_kernel_calc_single_layer;

        public: 
            OpenCLComputeDevice(cl::Device device);

        virtual std::unique_ptr<NetworkResourceHandle> RegisterNetwork(Network& network) override;

        virtual std::vector<float> Evaluate(const NetworkResourceHandle& network_handle, const std::span<float>& input) const override;

        static std::vector<cl::Device> GetDeviceList();

        static cl::Device AutoSelectDevice();
    };
}