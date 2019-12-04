using System;
using System.Collections.Generic;
using OpenCl.DotNetCore.Interop;
using OpenCl.DotNetCore.Interop.Devices;
using OpenCl.DotNetCore.Interop.Platforms;

namespace Macademy.OpenCL
{
    /// <summary>
    /// An OpenCL Computing device.
    /// </summary>
    public class ComputeDevice
    {
        IntPtr platform;
        IntPtr device;
        int platformId;
        int deviceId;

        public enum ComputeDeviceType { CPU, GPU, Accelerator, Unknown };

        internal ComputeDevice(IntPtr platform, IntPtr device, int platformId, int deviceId)
        {
            this.platform = platform;
            this.device = device;
            this.platformId = platformId;
            this.deviceId = deviceId;
        }

        internal IntPtr GetPlatform() { return platform; }

        internal IntPtr GetDevice() { return device; }


        /// <summary>
        /// Retrieves the specified information about the device.
        /// </summary>
        /// <typeparam name="T">The type of the data that is to be returned.</param>
        /// <param name="deviceInformation">The kind of information that is to be retrieved.</param>
        /// <exception cref="OpenClException">If the information could not be retrieved, then an <see cref="OpenClException"/> is thrown.</exception>
        /// <returns>Returns the specified information.</returns>
        private T GetDeviceInformation<T>(DeviceInformation deviceInformation)
        {
            // Retrieves the size of the return value in bytes, this is used to later get the full information
            UIntPtr returnValueSize;
            Result result = DevicesNativeApi.GetDeviceInformation(this.device, deviceInformation, UIntPtr.Zero, null, out returnValueSize);
            if (result != Result.Success)
                throw new OpenClException("The device information could not be retrieved.", result);
            
            // Allocates enough memory for the return value and retrieves it
            byte[] output = new byte[returnValueSize.ToUInt32()];
            result = DevicesNativeApi.GetDeviceInformation(this.device, deviceInformation, new UIntPtr((uint)output.Length), output, out returnValueSize);
            if (result != Result.Success)
                throw new OpenClException("The device information could not be retrieved.", result);

            // Returns the output
            return InteropConverter.To<T>(output);
        }

        /// <summary>
        /// Returns the type of the device (CPU, GPU, Accelerator)
        /// </summary>
        /// <returns>The type of the device, or Unknown if it cannot be determined</returns>
        public ComputeDeviceType GetDeviceType()
        {
            try {
                DeviceType result = (DeviceType)GetDeviceInformation<ulong>(DeviceInformation.Type);

                if ( ((int)result & (int)DeviceType.Accelerator) != 0 )
                    return ComputeDeviceType.Accelerator; 
                else if ( ((int)result & (int)DeviceType.Gpu) != 0 )
                    return ComputeDeviceType.GPU;
                else if ( ((int)result & (int)DeviceType.Cpu) != 0 )
                    return ComputeDeviceType.CPU;
            }
            catch (System.Exception) {
            }

            return ComputeDeviceType.Unknown;
        }

        /// <summary>
        /// The name of the device as visible from OpenCL 
        /// </summary>
        /// <returns>The device's name</returns>
        public String GetName()
        {
            try {
                return this.GetDeviceInformation<string>(DeviceInformation.Name);
            } catch (System.Exception) {
                return "unknown";
            }
        }

        /// <summary>
        /// The vendor of the device
        /// </summary>
        /// <returns>The vendor's name</returns>
        public String GetVendor()
        {
            try {
                return this.GetDeviceInformation<string>(DeviceInformation.Vendor);
            } catch (System.Exception) {
                return "unknown";
            }
        }

        /// <summary>
        /// The size of the global memory in bytes
        /// </summary>
        /// <returns>Global memory size in bytes</returns>
        public long GetGlobalMemorySize()
        {
            try {
                return (long)this.GetDeviceInformation<ulong>(DeviceInformation.GlobalMemorySize);
            } catch (System.Exception) {
                return 0L;
            }
        }

        /// <summary>
        /// Provides a list of available OpenCL devices on the system
        /// </summary>
        /// <returns>A list of OpenCL devices</returns>
        public static List<ComputeDevice> GetDevices()
        {
            List<ComputeDevice> ret = new List<ComputeDevice>();

            // Gets the number of available platforms
            uint numberOfAvailablePlatforms;
            Result result = PlatformsNativeApi.GetPlatformIds(0, null, out numberOfAvailablePlatforms);
            if (result != Result.Success)
                throw new OpenClException("The number of platforms could not be queried.", result);
            
            // Gets pointers to all the platforms
            IntPtr[] platformPointers = new IntPtr[numberOfAvailablePlatforms];
            result = PlatformsNativeApi.GetPlatformIds(numberOfAvailablePlatforms, platformPointers, out numberOfAvailablePlatforms);
            if (result != Result.Success)
                throw new OpenClException("The platforms could not be retrieved.", result);

            // Converts the pointers to platform objects
            int platformId = 0;
            foreach (IntPtr platformPointer in platformPointers){
                ++platformId;

                // Gets the number of available devices of the specified type
                uint numberOfAvailableDevices;
                Result result_d = DevicesNativeApi.GetDeviceIds(platformPointer, DeviceType.All, 0, null, out numberOfAvailableDevices);
                if (result_d != Result.Success)
                    throw new OpenClException("The number of available devices could not be queried.", result_d);

                // Gets the pointers to the devices of the specified type
                IntPtr[] devicePointers = new IntPtr[numberOfAvailableDevices];
                result_d = DevicesNativeApi.GetDeviceIds(platformPointer, DeviceType.All, numberOfAvailableDevices, devicePointers, out numberOfAvailableDevices);
                if (result_d != Result.Success)
                    throw new OpenClException("The devices could not be retrieved.", result_d);

                // Converts the pointer to device objects
                int deviceId = 0;
                foreach (IntPtr devicePointer in devicePointers){
                    ++deviceId;

                    ret.Add(new ComputeDevice(platformPointer, devicePointer, platformId, deviceId));
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
