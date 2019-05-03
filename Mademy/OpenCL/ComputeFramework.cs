using OpenCL.Net;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Mademy.OpenCL
{
    internal class ComputeFramework
    {
        public class MemoryAllocation
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

            ~MemoryAllocation()
            {
                Release();
            }

            public static MemoryAllocation CreateMemoryAllocation(Context clContext, int byteSize, MemFlags flags, IntPtr data)
            {
                ErrorCode err;
                var buffer = Cl.CreateBuffer(clContext, flags, new IntPtr(byteSize), data, out err);
                return new MemoryAllocation(buffer, byteSize, flags);
            }

            internal void Release()
            {
                if (buffer != null)
                {
                    Cl.ReleaseMemObject(buffer);
                    buffer = null;
                }
            }
        }

        private object lockObj = new object();
        private Context clContext;
        private ComputeDevice clDevice = null;
        private CommandQueue commandQueue;
        private Program clProgram;
        private Dictionary<String, Kernel> kernels = new Dictionary<string, Kernel>();
        private bool hasClInitialized = false;
        private List<MemoryAllocation> freeMemoryAllocations = new List<MemoryAllocation>();
        private List<MemoryAllocation> usedMemoryAllocations = new List<MemoryAllocation>();

        public ComputeDevice GetOpenCLDevice() { return clDevice; }

        public ComputeFramework(ComputeDevice clDevice, String[] kernelSource, String[] kernelNames, string compileArguments)
        {
            this.clDevice = clDevice;
            InitCL(kernelSource, kernelNames, compileArguments);
        }

        ~ComputeFramework() { CleanupCLResources(); }

        public void FlushWorkingCache()
        {
            foreach (var item in freeMemoryAllocations)
                item.Release();
            freeMemoryAllocations.Clear();

            foreach (var item in usedMemoryAllocations)
                item.Release();
            usedMemoryAllocations.Clear();
        }

        public void CleanupCLResources()
        {
            if (hasClInitialized)
            {
                FlushWorkingCache();

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

        public bool IsInitialized() { return hasClInitialized; }

        public void UploadToMemory(MemoryAllocation mem, int offset, int size, IntPtr data, bool IsBlocking)
        {
            Event e;
            Cl.EnqueueWriteBuffer(commandQueue, mem.buffer, IsBlocking ? Bool.True : Bool.False, new IntPtr(offset), new IntPtr(size), data, 0, null, out e);
        }

        public void UploadToMemory(MemoryAllocation mem, int offset, int[] data, bool IsBlocking)
        {
            Event e;
            unsafe
            {
                fixed (int* dataPtr = data)
                {
                    Cl.EnqueueWriteBuffer(commandQueue, mem.buffer, IsBlocking ? Bool.True : Bool.False, new IntPtr(offset), new IntPtr(data.Length * 4), new IntPtr(dataPtr), 0, null, out e);
                }
            }
        }

        public void UploadToMemory(MemoryAllocation mem, int offset, float[] data, bool IsBlocking)
        {
            Event e;
            unsafe
            {
                fixed (float* dataPtr = data)
                {
                    Cl.EnqueueWriteBuffer(commandQueue, mem.buffer, IsBlocking ? Bool.True : Bool.False, new IntPtr(offset), new IntPtr(data.Length * 4), new IntPtr(dataPtr), 0, null, out e);
                }
            }
        }

        public MemoryAllocation GetMemoryFor(MemFlags flags, int[] data)
        {
            unsafe
            {
                fixed (int* dataPtr = data)
                {
                    return GetMemoryFor(data.Length * 4, flags, new IntPtr( dataPtr));
                }
            }
        }

        public MemoryAllocation GetMemoryFor(MemFlags flags, float[] data)
        {
            unsafe
            {
                fixed (float* dataPtr = data)
                {
                    return GetMemoryFor(data.Length * 4, flags, new IntPtr(dataPtr));
                }
            }
        }

        public MemoryAllocation GetMemoryFor(int requiredSizeInBytes, MemFlags flags, IntPtr data)
        {
            if (!IsInitialized())
                return null;

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

        public void UnuseMemoryAllocations()
        {
            foreach (var item in usedMemoryAllocations)
            {
                freeMemoryAllocations.Add(item);
            }
            usedMemoryAllocations.Clear();
        }

        public void SetKernelArg(string kernelName, uint idx, MemoryAllocation mem)
        {
            Cl.SetKernelArg(kernels[kernelName], idx, mem.buffer);
        }

        public void ReadBuffer(MemoryAllocation mem, bool isBlocking, IntPtr offset, IntPtr lengthInBytes, IntPtr output)
        {
            Event ev;
            Cl.EnqueueReadBuffer(commandQueue, mem.buffer, isBlocking ? Bool.True : Bool.False, offset, lengthInBytes, output, 0, null, out ev);
        }

        public void EnqueueKernel(string kernelName, IntPtr[] globalWorkSize, IntPtr[] localWorkSize)
        {
            Event ev;
            Cl.EnqueueNDRangeKernel(commandQueue, kernels[kernelName], (uint)globalWorkSize.Length, null, globalWorkSize, localWorkSize, 0, null, out ev);
        }

        private void InitCL(String[] kernelSource, String[] kernelNames, string compileArguments)
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

                    clProgram = Cl.CreateProgramWithSource(clContext, 1, kernelSource.ToArray(), null, out err);
                    if (err != ErrorCode.Success) throw new Exception("Failed to create program! " + err.ToString());

                    err = Cl.BuildProgram(clProgram, 1, new Device[] { clDevice.GetDevice() }, compileArguments, null, IntPtr.Zero);
                    if (err != ErrorCode.Success)
                    {
                        var infoBuffer = Cl.GetProgramBuildInfo(clProgram, clDevice.GetDevice(), ProgramBuildInfo.Log, out err);
                        throw new Exception("Failed to build program! " + err.ToString() + " " + infoBuffer.ToString());
                    }

                    foreach (var item in kernelNames)
                    {
                        kernels[item] = Cl.CreateKernel(clProgram, item, out err);
                    }

                    if (err != ErrorCode.Success) throw new Exception("Failed to create compute kernel! " + err.ToString());

                    hasClInitialized = true;
                }
            }
        }

    }
}
