using Mademy;
using OpenCL.Net;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Mademy
{
    public class MathLib
    {
        private class MemoryAllocation
        {
            public IMem buffer;
            public int bufferSizeInBytes;
            public MemFlags flags;

            public MemoryAllocation(IMem buffer, int size, MemFlags flags)
            {
                this.buffer = buffer;
                this.bufferSizeInBytes = size;
                this.flags = flags;
            }

            public static MemoryAllocation CreateMemoryAllocation(Context clContext, int byteSize, MemFlags flags, IntPtr data)
            {
                ErrorCode err;
                var buffer = Cl.CreateBuffer(clContext, flags, new IntPtr(byteSize), data, out err);
                return new MemoryAllocation(buffer, byteSize, flags);
            }

            internal void Release()
            {
                Cl.ReleaseMemObject(buffer);
            }
        }

        private static string calcLayerKernel = "calcLayer";

        private object lockObj = new object();
        Context clContext;
        ComputeDevice clDevice = null;
        CommandQueue commandQueue;
        Program clProgram;
        Dictionary<String, Kernel> kernels = new Dictionary<string, Kernel>();
        bool hasClInitialized = false;
        List<MemoryAllocation> freeMemoryAllocations = new List<MemoryAllocation>();
        List<MemoryAllocation> usedMemoryAllocations = new List<MemoryAllocation>();

        public MathLib(ComputeDevice clDevice = null)
        {
            this.clDevice = clDevice;
            InitCL();
        }

        ~MathLib() { CleanupCLResources(); }



        private void CleanupCLResources()
        {
            if (hasClInitialized)
            {
                foreach (var item in freeMemoryAllocations)
                    item.Release();

                foreach (var item in kernels)
                {
                    Cl.ReleaseKernel(item.Value);
                }
                kernels.Clear();

                Cl.ReleaseProgram(clProgram);
                Cl.ReleaseCommandQueue(commandQueue);
                Cl.ReleaseContext(clContext);

                hasClInitialized = false;
            }
        }

        public MathLib Clone()
        {
            return new MathLib(this.clDevice);
        }

        private MemoryAllocation GetMemoryFor(int requiredSizeInBytes, MemFlags flags, IntPtr data)
        {
            MemoryAllocation candidate = null;
            int bestMatchingSize = int.MaxValue;
            int itemToSwap = -1;

            for (int i = 0; i < freeMemoryAllocations.Count; i++)
            {
                var item = freeMemoryAllocations[i];
                if (item.flags == flags && item.bufferSizeInBytes >= requiredSizeInBytes && item.bufferSizeInBytes < bestMatchingSize) //Select the smallest sufficient memory allocation from our allocations
                {
                    bestMatchingSize = item.bufferSizeInBytes;
                    candidate = item;
                    itemToSwap = i;
                    if (item.bufferSizeInBytes == requiredSizeInBytes)
                        break;
                }
            }

            if (candidate == null)
            {
                candidate = MemoryAllocation.CreateMemoryAllocation(clContext, requiredSizeInBytes, flags, data);
                usedMemoryAllocations.Add(candidate);
            }
            else
            {
                freeMemoryAllocations.RemoveAt(itemToSwap);
                usedMemoryAllocations.Add(candidate);
                if (flags.HasFlag(MemFlags.CopyHostPtr))
                {
                    Event e;
                    Cl.EnqueueWriteBuffer(commandQueue, candidate.buffer, Bool.False, IntPtr.Zero, new IntPtr(requiredSizeInBytes), data, 0, null, out e);
                }
            }

            return candidate;
        }

        void UnuseMemoryAllocations()
        {
            foreach (var item in usedMemoryAllocations)
            {
                freeMemoryAllocations.Add(item);
            }
            usedMemoryAllocations.Clear();
        }

        private void InitCL()
        {
            lock (lockObj)
            {
                if (!hasClInitialized && clDevice != null)
                {
                    ErrorCode err;
                    var devicesArray = new Device[] { clDevice.GetDevice() };
                    clContext = Cl.CreateContext(null, 1, devicesArray, null, IntPtr.Zero, out err);
                    if (err != ErrorCode.Success) throw new Exception("Failed to create context! " + err.ToString());

                    commandQueue = Cl.CreateCommandQueue(clContext, clDevice.GetDevice(), CommandQueueProperties.None, out err);
                    if (err != ErrorCode.Success) throw new Exception("Failed to create command queue! " + err.ToString());

                    clProgram = Cl.CreateProgramWithSource(clContext, 1, new String[] { CLSourceProvider.ReadSourceFile() }, null, out err);
                    if (err != ErrorCode.Success) throw new Exception("Failed to create program! " + err.ToString());

                    err = Cl.BuildProgram(clProgram, 1, new Device[] { clDevice.GetDevice() }, "-cl-finite-math-only -Werror", null, IntPtr.Zero);
                    if (err != ErrorCode.Success)
                    {
                        var infoBuffer = Cl.GetProgramBuildInfo(clProgram, clDevice.GetDevice(), ProgramBuildInfo.Log, out err);
                        throw new Exception("Failed to build program! " + err.ToString() + " " + infoBuffer.ToString());
                    }

                    kernels[calcLayerKernel] = Cl.CreateKernel(clProgram, calcLayerKernel, out err);
                    if (err != ErrorCode.Success) throw new Exception("Failed to create compute kernel! " + err.ToString());

                    hasClInitialized = true;
                }
            }
        }

        public float[] CalculateLayer(float[,] weightMx, float[] bias, float[] prevActivations, IActivationFunction sigmoidFunction)
        {
            if (weightMx.GetLength(1) != prevActivations.GetLength(0))
                throw new Exception("Invalid input");

            if (!hasClInitialized) //CPU fallback
            {
                float[] ret = new float[weightMx.GetLength(0)];
                for (int m = 0; m < weightMx.GetLength(0); m++)
                {
                    float acc = 0.0f;
                    for (int k = 0; k < weightMx.GetLength(1); k++)
                    {
                        acc += weightMx[m, k] * prevActivations[k];
                    }
                    acc += bias[m];

                    ret[m] = sigmoidFunction.Calculate(acc);
                }
                return ret;
            }

            int matrixRows = weightMx.GetLength(0);

            ErrorCode err;
            MemoryAllocation mem_param_weightMx, mem_param_bias, mem_param_prevActivation, mem_param_config, mem_param_output;
            unsafe
            {
                fixed (float* weightArrayPtr = weightMx)
                {
                    mem_param_weightMx = GetMemoryFor(weightMx.Length * 4, MemFlags.ReadOnly | MemFlags.CopyHostPtr, new IntPtr(weightArrayPtr));
                }
                fixed (float* biasPtr = bias)
                {
                    mem_param_bias = GetMemoryFor(bias.Length * 4, MemFlags.ReadOnly | MemFlags.CopyHostPtr, new IntPtr(biasPtr));
                }
                fixed (float* prevActivationPtr = prevActivations)
                {
                    mem_param_prevActivation = GetMemoryFor(prevActivations.Length * 4, MemFlags.ReadOnly | MemFlags.CopyHostPtr, new IntPtr(prevActivationPtr));
                }

                int[] configParams = new int[] { /*rows: */weightMx.GetLength(0), /*cols: */weightMx.GetLength(1), /*ApplySigmoid*/ sigmoidFunction.GetOpenCLFunctionId() };
                fixed (int* configPtr = configParams)
                {
                    mem_param_config = GetMemoryFor(configParams.Length * 4, MemFlags.ReadOnly | MemFlags.CopyHostPtr, new IntPtr(configPtr));
                }
                mem_param_output = GetMemoryFor(matrixRows * 4, MemFlags.WriteOnly, IntPtr.Zero);
            }

            var kernel = kernels[calcLayerKernel];
            Cl.SetKernelArg(kernel, 0, mem_param_weightMx.buffer);
            Cl.SetKernelArg(kernel, 1, mem_param_bias.buffer);
            Cl.SetKernelArg(kernel, 2, mem_param_prevActivation.buffer);
            Cl.SetKernelArg(kernel, 3, mem_param_config.buffer);
            Cl.SetKernelArg(kernel, 4, mem_param_output.buffer);

            Event ev;
            int localWorkgroupSize = 32;
            int globalWorkSize = (matrixRows % localWorkgroupSize == 0) ? matrixRows : (matrixRows + (localWorkgroupSize - (matrixRows % localWorkgroupSize)));
            Cl.EnqueueNDRangeKernel(commandQueue, kernel, 1, null, new IntPtr[] { new IntPtr(globalWorkSize) }, new IntPtr[] { new IntPtr(localWorkgroupSize) }, 0, null, out ev);

            float[] output = new float[matrixRows];
            Cl.EnqueueReadBuffer(commandQueue, mem_param_output.buffer, Bool.True, 0, matrixRows, output, 0, null, out ev);

            UnuseMemoryAllocations();

            return output;
        }
        
    }
}
