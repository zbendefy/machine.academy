using System;
using System.Collections.Generic;
using System.Linq;
using OpenCl.DotNetCore.Interop;
using OpenCl.DotNetCore.Interop.Devices;
using OpenCl.DotNetCore.Interop.Memory;
using OpenCl.DotNetCore.Interop.Platforms;
using static Macademy.OpenCL.ComputeFramework;

namespace Macademy.OpenCL
{
    internal class OpenCLComputeDeviceDesc : ComputeDeviceDesc
    {
        internal IntPtr platform;
        internal IntPtr device;
        internal int platformId;
        internal int deviceId;

        public override ComputeDevice CreateDevice()
        {
            return OpenCLDevice.CreateDevice(this);
        }

        public override string GetDeviceAccessType()
        {
            return "OpenCL";
        }

        public override int GetDeviceCoreCount()
        {
            return OpenCLDevice.GetDeviceInformation<int>(device, DeviceInformation.MaximumComputeUnits);
        }

        public override long GetDeviceMemorySize()
        {
            return (long)OpenCLDevice.GetDeviceInformation<ulong>(device, DeviceInformation.GlobalMemorySize);
        }

        public override string GetDeviceName()
        {
            return OpenCLDevice.GetDeviceInformation<string>(device, DeviceInformation.Name);
        }
    }

    /// <summary>
    /// An OpenCL Computing device.
    /// </summary>
    internal class OpenCLDevice : ComputeDevice
    {
        IntPtr platform;
        IntPtr device;
        int platformId;
        int deviceId;

        private ComputeFramework computeFramework = null;
        private DeviceConfig deviceConfig;
        private static readonly string calcLayerKernel = "calcSingleLayer";
        private static readonly string forwardPass = "trainingForwardPass";
        private static readonly string backwardPassKernel = "trainingBackwardPass";

        /// <summary>
        /// A hardware specific configuration providing configurable parameters for an OpenCL device
        /// </summary>
        public class DeviceConfig
        {
            public readonly int idealWorkgroupSizeX = 8;
            public readonly int idealWorkgroupSizeY = 8;
            public readonly string compileOptions = "-cl-mad-enable -cl-no-signed-zeros";

            public DeviceConfig() { }

            public DeviceConfig(int idealWorkgroupSizeX, int idealWorkgroupSizeY, string compileOptions)
            {
                this.idealWorkgroupSizeX = idealWorkgroupSizeX;
                this.idealWorkgroupSizeY = idealWorkgroupSizeY;
                this.compileOptions = compileOptions;
            }
        }

        internal static ComputeDevice CreateDevice(OpenCLComputeDeviceDesc desc, DeviceConfig deviceConfig = null)
        {
            return new OpenCLDevice(desc.platform, desc.device, desc.platformId, desc.deviceId, deviceConfig);
        }

        public enum ComputeDeviceType { CPU, GPU, Accelerator, Unknown };

        private int ExtendGlobalWorkSize(int desiredGlobalSize, int localSize)
        {
            return ((desiredGlobalSize % localSize) == 0) ? desiredGlobalSize : (desiredGlobalSize + (localSize - (desiredGlobalSize % localSize)));
        }

        internal OpenCLDevice(IntPtr platform, IntPtr device, int platformId, int deviceId, DeviceConfig _deviceConfig)
        {
            this.platform = platform;
            this.device = device;
            this.platformId = platformId;
            this.deviceId = deviceId;
            this.deviceConfig = _deviceConfig == null ? new DeviceConfig() : _deviceConfig;

            computeFramework = new ComputeFramework(GetDevice(), new string[] { CLSourceProvider.ReadSourceFile() }, new string[] { calcLayerKernel, forwardPass, backwardPassKernel }, deviceConfig.compileOptions);
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
        internal static T GetDeviceInformation<T>(IntPtr device, DeviceInformation deviceInformation)
        {
            // Retrieves the size of the return value in bytes, this is used to later get the full information
            UIntPtr returnValueSize;
            Result result = DevicesNativeApi.GetDeviceInformation(device, deviceInformation, UIntPtr.Zero, null, out returnValueSize);
            if (result != Result.Success)
                throw new OpenClException("The device information could not be retrieved.", result);
            
            // Allocates enough memory for the return value and retrieves it
            byte[] output = new byte[returnValueSize.ToUInt32()];
            result = DevicesNativeApi.GetDeviceInformation(device, deviceInformation, new UIntPtr((uint)output.Length), output, out returnValueSize);
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
                DeviceType result = (DeviceType)GetDeviceInformation<ulong>(device, DeviceInformation.Type);

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
        public override string GetName()
        {
            try {
                return GetDeviceInformation<string>(device, DeviceInformation.Name);
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
                return GetDeviceInformation<string>(device, DeviceInformation.Vendor);
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
                return (long)GetDeviceInformation<ulong>(device, DeviceInformation.GlobalMemorySize);
            } catch (System.Exception) {
                return 0L;
            }
        }

        /// <summary>
        /// Provides a list of available OpenCL devices on the system
        /// </summary>
        /// <returns>A list of OpenCL devices</returns>
        public static List<ComputeDeviceDesc> GetDevices()
        {
            List<ComputeDeviceDesc> ret = new List<ComputeDeviceDesc>();

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

                    OpenCLComputeDeviceDesc devDesc = new OpenCLComputeDeviceDesc();
                    devDesc.platform = platformPointer;
                    devDesc.device = devicePointer;
                    devDesc.platformId= platformId;
                    devDesc.deviceId = deviceId;
                    ret.Add(devDesc);
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

        public override string GetDeviceAccessMode()
        {
            return "OpenCL";
        }

        public override int GetDeviceCoreCount()
        {
            try
            {
                return GetDeviceInformation<int>(device, DeviceInformation.MaximumComputeUnits);
            }
            catch (System.Exception)
            {
                return 0;
            }
        }


        public override void FlushWorkingCache()
        {
            computeFramework.FlushWorkingCache();
        }


        public override unsafe float[] CalculateLayer(float[,] weightMx, float[] bias, float[] prevActivations, IActivationFunction sigmoidFunction)
        {
            int matrixRows = weightMx.GetLength(0);
            float[] output = new float[matrixRows];
            int[] configParams = new int[] { /*rows: */weightMx.GetLength(0), /*cols: */weightMx.GetLength(1), /*ApplySigmoid*/ sigmoidFunction.GetOpenCLFunctionId() };

            fixed (int* configPtr = configParams)
            {
                fixed (float* weightArrayPtr = weightMx, biasPtr = bias, prevActivationPtr = prevActivations)
                {
                    MemoryAllocation mem_param_weightMx, mem_param_bias, mem_param_prevActivation, mem_param_config, mem_param_output;
                    mem_param_weightMx = computeFramework.GetMemoryFor(weightMx.Length * 4, MemoryFlag.ReadOnly | MemoryFlag.CopyHostPointer, new IntPtr(weightArrayPtr));
                    mem_param_bias = computeFramework.GetMemoryFor(bias.Length * 4, MemoryFlag.ReadOnly | MemoryFlag.CopyHostPointer, new IntPtr(biasPtr));
                    mem_param_prevActivation = computeFramework.GetMemoryFor(prevActivations.Length * 4, MemoryFlag.ReadOnly | MemoryFlag.CopyHostPointer, new IntPtr(prevActivationPtr));
                    mem_param_config = computeFramework.GetMemoryFor(configParams.Length * 4, MemoryFlag.ReadOnly | MemoryFlag.CopyHostPointer, new IntPtr(configPtr));
                    mem_param_output = computeFramework.GetMemoryFor(matrixRows * 4, MemoryFlag.WriteOnly, IntPtr.Zero);

                    computeFramework.SetKernelArg(calcLayerKernel, 0, mem_param_weightMx);
                    computeFramework.SetKernelArg(calcLayerKernel, 1, mem_param_bias);
                    computeFramework.SetKernelArg(calcLayerKernel, 2, mem_param_prevActivation);
                    computeFramework.SetKernelArg(calcLayerKernel, 3, mem_param_config);
                    computeFramework.SetKernelArg(calcLayerKernel, 4, mem_param_output);

                    int localWorkgroupSize = 32;
                    int globalWorkSize = ExtendGlobalWorkSize(matrixRows, localWorkgroupSize);
                    computeFramework.EnqueueKernel(calcLayerKernel, new IntPtr[] { new IntPtr(globalWorkSize) }, new IntPtr[] { new IntPtr(localWorkgroupSize) });


                    fixed (float* outputPtr = output)
                    {
                        computeFramework.ReadBuffer(mem_param_output, true, UIntPtr.Zero, new UIntPtr((uint)matrixRows * 4U), new IntPtr(outputPtr));
                    }
                }
            }

            computeFramework.UnuseMemoryAllocations();

            return output;
        }

        public override void Uninitialize()
        {
            computeFramework.CleanupCLResources();
        }

        public override unsafe List<List<NeuronData>> CalculateAccumulatedGradientForMinibatch(Network network, TrainingSuite suite, int trainingDataBegin, int trainingDataEnd)
        {
            int trainingSamples = trainingDataEnd - trainingDataBegin;
            var ret = Utils.CreateGradientVector(network);

            int[] networkConfigParams = null;
            int totalWeightAndBiasCount = 0;
            int delta_k_vectorSize = 0;
            int totalActivationCount = 0; //Add 
            int inputActivationCount = network.layers.First().GetWeightsPerNeuron();
            {
                foreach (var item in network.layers)
                {
                    totalActivationCount += item.GetNeuronCount();
                }

                List<int> networkConfigParamsList = new List<int>();
                //0
                networkConfigParamsList.Add(0); //layer index to be processed
                //1
                networkConfigParamsList.Add(network.layers.Count); //Layer count
                //2
                networkConfigParamsList.Add(trainingSamples); //numTrainingSamples
                //3
                networkConfigParamsList.Add(network.activationFunction.GetOpenCLFunctionId()); //Activation function
                //4
                networkConfigParamsList.Add(suite.config.costFunction.GetOpenCLFunctionID()); //Cost function
                //5
                networkConfigParamsList.Add(totalActivationCount); //totalActivationCount
                //6
                networkConfigParamsList.Add(0); //totalWeightsAndBiases
                //7
                networkConfigParamsList.Add(0); //widestLayerNeuronCount
                //8
                networkConfigParamsList.Add(network.layers.First().GetWeightsPerNeuron()); //Input count
                for (int i = 0; i < network.layers.Count; i++)
                {
                    networkConfigParamsList.Add(network.layers[i].GetNeuronCount()); //Layer neuron count
                    totalWeightAndBiasCount += network.layers[i].biases.Length;
                    totalWeightAndBiasCount += network.layers[i].weightMx.Length;
                    if (i > 0) //The first layer will not write the delta_k vector, so it shouldn't contribute to its size.
                        delta_k_vectorSize = Math.Max(network.layers[i].GetNeuronCount(), delta_k_vectorSize);
                }

                networkConfigParamsList[6] = totalWeightAndBiasCount;
                networkConfigParamsList[7] = delta_k_vectorSize;

                networkConfigParams = networkConfigParamsList.ToArray();
            }

            float[] desiredOutputs = new float[network.layers.Last().GetNeuronCount() * trainingSamples];
            float[] outputGradient = new float[totalWeightAndBiasCount];//Memory layout is: [weights, biases for trainingsample0, layer0-N][weights, biases for trainingsample1, layer0-N] ...
            float[] inputParameters = new float[trainingSamples * inputActivationCount];
            float[] weightsAndBiases = new float[totalWeightAndBiasCount];

            fixed (int* networkConfigParamsPtr = networkConfigParams)
            {
                fixed (float* outputGradientPtr = outputGradient, desiredOutputsPtr = desiredOutputs, inputParametersPtr = inputParameters, weightsAndBiasesPtr = weightsAndBiases)
                {
                    MemoryAllocation mem_NetworkConfigParams = computeFramework.GetMemoryFor(networkConfigParams.Length * 4, MemoryFlag.ReadOnly | MemoryFlag.CopyHostPointer, new IntPtr(networkConfigParamsPtr));

                    for (int i = 0; i < trainingSamples; ++i)
                        Buffer.BlockCopy(suite.trainingData[trainingDataBegin + i].input, 0, inputParameters, i * inputActivationCount * 4, inputActivationCount * 4);
                    MemoryAllocation mem_InputActivations = computeFramework.GetMemoryFor(inputParameters.Length * 4, MemoryFlag.ReadOnly | MemoryFlag.CopyHostPointer, new IntPtr(inputParametersPtr));

                    ///Contains the whole network's activation values, and Z values for each training sample
                    ///Memory layout for one layer is like this: [...input values...][...first layer's activations...][...second layer's activations]...[last layer's activations][first layer's z values][second layer's zvalues]...[last layer's z values]
                    ///After that, the next layer's same values are there
                    MemoryAllocation mem_activationsAndZValues = computeFramework.GetMemoryFor(totalActivationCount * trainingSamples * 2 * 4, MemoryFlag.ReadWrite, IntPtr.Zero);

                    {
                        int offset = 0;
                        foreach (var layer in network.layers)
                        {
                            Buffer.BlockCopy(layer.weightMx, 0, weightsAndBiases, offset, layer.weightMx.Length * 4);
                            offset += layer.weightMx.Length * 4;
                            Buffer.BlockCopy(layer.biases, 0, weightsAndBiases, offset, layer.biases.Length * 4);
                            offset += layer.biases.Length * 4;
                        }
                    }
                    MemoryAllocation mem_weightsAndBiases = computeFramework.GetMemoryFor(weightsAndBiases.Length * 4, MemoryFlag.ReadOnly | MemoryFlag.CopyHostPointer, new IntPtr(weightsAndBiasesPtr));

                    //delta_k_vector is double buffered (hence the * 2). In a pass, the previous delta_k values are read, and the next ones are written
                    //Memory layout is: [delta_k_vector buffer1 of trainingSample0][delta_k_vector buffer2 of trainingSample0] [delta_k_vector buffer1 of trainingSample1][delta_k_vector buffer2 of trainingSample1] ...
                    MemoryAllocation mem_delta_k_vector = computeFramework.GetMemoryFor(Math.Max(1, delta_k_vectorSize * trainingSamples * 2 * 4), MemoryFlag.ReadWrite, IntPtr.Zero);

                    computeFramework.SetKernelArg(forwardPass, 0, mem_NetworkConfigParams);
                    computeFramework.SetKernelArg(forwardPass, 1, mem_activationsAndZValues);
                    computeFramework.SetKernelArg(forwardPass, 2, mem_InputActivations);
                    computeFramework.SetKernelArg(forwardPass, 3, mem_weightsAndBiases);

                    int[] layerIndexBuffer = new int[network.layers.Count];
                    for (int i = 0; i < layerIndexBuffer.Length; ++i)
                        layerIndexBuffer[i] = i;

                    var localWorkGroupSize = new IntPtr[] { new IntPtr(deviceConfig.idealWorkgroupSizeX), new IntPtr(deviceConfig.idealWorkgroupSizeY) };
                    var globalWorkSize = new IntPtr[] { new IntPtr(0), new IntPtr(ExtendGlobalWorkSize(trainingSamples, localWorkGroupSize[1].ToInt32())) };

                    #region Forward pass
                    for (int i = 0; i < network.layers.Count; ++i)
                    {
                        if (i > 0)
                        {
                            computeFramework.UploadToMemory(mem_NetworkConfigParams, i, layerIndexBuffer, false, 1); //Update layer index to be processed by the kernel
                        }

                        globalWorkSize[0] = new IntPtr(ExtendGlobalWorkSize(network.layers[i].GetNeuronCount(), localWorkGroupSize[0].ToInt32()));
                        computeFramework.EnqueueKernel(forwardPass, globalWorkSize, localWorkGroupSize);
                        // todo: run forward pass
                    }
                    #endregion

                    #region backward pass

                    int desiredOutputByteSizePerTrainigSample = network.layers.Last().GetNeuronCount() * 4;
                    for (int i = 0; i < trainingSamples; ++i)
                        Buffer.BlockCopy(suite.trainingData[trainingDataBegin + i].desiredOutput, 0, desiredOutputs, i * desiredOutputByteSizePerTrainigSample, desiredOutputByteSizePerTrainigSample);
                    var mem_desired_outputs = computeFramework.GetMemoryFor(desiredOutputs.Length * 4, MemoryFlag.ReadOnly | MemoryFlag.CopyHostPointer, new IntPtr(desiredOutputsPtr));

                    var mem_param_gradient = computeFramework.GetMemoryFor(outputGradient.Length * 4, MemoryFlag.ReadWrite | MemoryFlag.CopyHostPointer, new IntPtr(outputGradientPtr));

                    computeFramework.SetKernelArg(backwardPassKernel, 0, mem_NetworkConfigParams);
                    computeFramework.SetKernelArg(backwardPassKernel, 1, mem_activationsAndZValues);
                    computeFramework.SetKernelArg(backwardPassKernel, 2, mem_delta_k_vector);
                    computeFramework.SetKernelArg(backwardPassKernel, 3, mem_param_gradient);
                    computeFramework.SetKernelArg(backwardPassKernel, 4, mem_desired_outputs);
                    computeFramework.SetKernelArg(backwardPassKernel, 5, mem_InputActivations);
                    computeFramework.SetKernelArg(backwardPassKernel, 6, mem_weightsAndBiases);

                    //Run backward pass for all hidden layers
                    for (int i = network.layers.Count - 1; i >= 0; --i)
                    {
                        globalWorkSize[0] = new IntPtr(ExtendGlobalWorkSize(network.layers[i].GetNeuronCount(), localWorkGroupSize[0].ToInt32()));
                        if (i != network.layers.Count - 1)
                            computeFramework.UploadToMemory(mem_NetworkConfigParams, i, layerIndexBuffer, false, 1); //Update layer index to be processed by the kernel
                        computeFramework.EnqueueKernel(backwardPassKernel, globalWorkSize, localWorkGroupSize);
                    }
                    #endregion

                    computeFramework.FlushCommandBuffer();

                    computeFramework.ReadBuffer(mem_param_gradient, true, new UIntPtr(0), new UIntPtr(mem_param_gradient.bufferSizeInBytes), new IntPtr(outputGradientPtr));
                }
            }

            computeFramework.UnuseMemoryAllocations();

            int gradIdx = 0;
            foreach (var layer in ret)
            {
                foreach (var neuron in layer)
                {
                    Buffer.BlockCopy(outputGradient, gradIdx * 4, neuron.weights, 0, neuron.weights.Length * 4);
                    gradIdx += neuron.weights.Length;
                    neuron.bias = outputGradient[gradIdx];
                    ++gradIdx;
                }
            }

            return ret;
        }

    }
}
