using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCL.Net;

namespace CLMath
{
    public class ComputeDevice
    {
        Platform platform;
        Device device;
        int platformId;
        int deviceId;

        public enum ComputeDeviceType { CPU, GPU, Accelerator, Unknown };

        public ComputeDevice(Platform platform, Device device, int platformId, int deviceId)
        {
            this.platform = platform;
            this.device = device;
            this.platformId = platformId;
            this.deviceId = deviceId;
        }

        public Platform GetPlatform() { return platform; }
        public Device GetDevice() { return device; }

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

        public String GetName()
        {
            ErrorCode err;
            var result = Cl.GetDeviceInfo(device, DeviceInfo.Name, out err);
            if (err != ErrorCode.Success)
                return "unknown";
            return result.ToString();
        }

        public String GetVendor()
        {
            ErrorCode err;
            var result = Cl.GetDeviceInfo(device, DeviceInfo.Vendor, out err);
            if (err != ErrorCode.Success)
                return "unknown";
            return result.ToString();
        }

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

        public int GetPlatformID()
        {
            return platformId;
        }

        public int GetDeviceID()
        {
            return deviceId;
        }

        public override String ToString() { return GetName(); }
    }
}
