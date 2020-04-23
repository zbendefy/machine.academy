using System;
using System.Collections.Generic;
using System.Text;

namespace Macademy
{
    public static class ComputeDeviceFactory
    {
        public static ComputeDevice GetComputeDeviceById(int id)
        {
            return CreateComputeDevice(GetComputeDevices()[id]);
        }

        public static List<ComputeDeviceDesc> GetComputeDevices()
        {
            List<ComputeDeviceDesc> ret = new List<ComputeDeviceDesc>();

            ret.AddRange(CPUComputeDevice.GetDevices());
            ret.AddRange(OpenCL.OpenCLDevice.GetDevices());

            return ret;
        }

        public static ComputeDevice CreateComputeDevice(ComputeDeviceDesc desc)
        {
            return desc == null ? null : desc.CreateDevice();
        }

        public static ComputeDevice CreateFallbackComputeDevice()
        {
            return CreateComputeDevice(CPUComputeDevice.GetDevices()[0]);
        }
    }
}
