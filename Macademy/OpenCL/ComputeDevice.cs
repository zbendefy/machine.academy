using System;
using System.Collections.Generic;
using OpenCL.Net;

namespace Macademy.OpenCL
{
    /// <summary>
    /// An OpenCL Computing device.
    /// </summary>
    public class ComputeDevice
    {
        Platform platform;
        Device device;
        int platformId;
        int deviceId;

        public enum ComputeDeviceType { CPU, GPU, Accelerator, Unknown };

        internal ComputeDevice(Platform platform, Device device, int platformId, int deviceId)
        {
            this.platform = platform;
            this.device = device;
            this.platformId = platformId;
            this.deviceId = deviceId;
        }

        internal Platform GetPlatform() { return platform; }

        internal Device GetDevice() { return device; }

        /// <summary>
        /// Returns the type of the device (CPU, GPU, Accelerator)
        /// </summary>
        /// <returns>The type of the device, or Unknown if it cannot be determined</returns>
        public ComputeDeviceType GetDeviceType()
        {
            ErrorCode err;
            var result = Cl.GetDeviceInfo(device, DeviceInfo.Type, out err);
            if (err == ErrorCode.Success)
            {
                int type = result.CastTo<int>();
                switch (type)
                {
                    case (int)DeviceType.Cpu:
                        return ComputeDeviceType.CPU;
                    case (int)DeviceType.Gpu:
                        return ComputeDeviceType.GPU;
                    case (int)DeviceType.Accelerator:
                        return ComputeDeviceType.Accelerator;
                    default:
                        break;
                }
            }
            return ComputeDeviceType.Unknown;
        }

        /// <summary>
        /// The name of the device as visible from OpenCL 
        /// </summary>
        /// <returns>The device's name</returns>
        public String GetName()
        {
            ErrorCode err;
            var result = Cl.GetDeviceInfo(device, DeviceInfo.Name, out err);
            if (err != ErrorCode.Success)
                return "unknown";
            return result.ToString();
        }

        /// <summary>
        /// The vendor of the device
        /// </summary>
        /// <returns>The vendor's name</returns>
        public String GetVendor()
        {
            ErrorCode err;
            var result = Cl.GetDeviceInfo(device, DeviceInfo.Vendor, out err);
            if (err != ErrorCode.Success)
                return "unknown";
            return result.ToString();
        }

        /// <summary>
        /// The size of the global memory in bytes
        /// </summary>
        /// <returns>Global memory size in bytes</returns>
        public long GetGlobalMemorySize()
        {
            ErrorCode err;
            var result = Cl.GetDeviceInfo(device, DeviceInfo.GlobalMemSize, out err);
            if (err != ErrorCode.Success)
                return 0;
            return result.CastTo<long>();
        }

        /// <summary>
        /// Provides a list of available OpenCL devices on the system
        /// </summary>
        /// <returns>A list of OpenCL devices</returns>
        public static List<ComputeDevice> GetDevices()
        {
            List<ComputeDevice> ret = new List<ComputeDevice>();

            ErrorCode err;
            Platform[] platforms = Cl.GetPlatformIDs(out err);

            if (err != ErrorCode.Success)
                return ret;

            for (int i = 0; i < platforms.Length; i++)
            {
                Device[] devices = Cl.GetDeviceIDs(platforms[i], DeviceType.All, out err);

                if (err != ErrorCode.Success)
                    continue;

                for (int j = 0; j < devices.Length; j++)
                {
                    ret.Add(new ComputeDevice(platforms[i], devices[j], i, j));
                }
            }

            return ret;
        }

        /// <summary>
        /// The OpenCL platform id
        /// </summary>
        /// <returns>The OpenCL platform id</returns>
        public int GetPlatformID()
        {
            return platformId;
        }

        /// <summary>
        /// The OpenCL device id in the platform it is in
        /// </summary>
        /// <returns>The OpenCL device id</returns>
        public int GetDeviceID()
        {
            return deviceId;
        }

        public override String ToString() { return GetName(); }
    }
}
