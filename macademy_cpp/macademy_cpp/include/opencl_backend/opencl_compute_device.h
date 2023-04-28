#pragma once

#include "i_compute_backend.h"
#include "opencl_common.h"

namespace macademy
{
    class OpenCLComputeDevice : public IComputeDevice
    {
        cl::Device m_device;
        cl::Context m_context;
        mutable cl::CommandQueue m_command_queue;
        cl::Program m_program;

        mutable std::unique_ptr<cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>> m_kernel_calc_single_layer;
        cl::size_type m_kernel_calc_single_layer_ideal_workgroup_size = 32;

        public: 
            OpenCLComputeDevice(cl::Device device);

        virtual std::unique_ptr<NetworkResourceHandle> RegisterNetwork(Network& network) override;

        virtual std::vector<float> Evaluate(const NetworkResourceHandle& network_handle, const std::span<float>& input) const override;

        static std::vector<cl::Device> GetDeviceList();

        static cl::Device AutoSelectDevice();

        std::string GetDeviceName() const override;

        size_t GetTotalMemory() const override;

        uint32_t GetComputeUnits() const override;
    };
}