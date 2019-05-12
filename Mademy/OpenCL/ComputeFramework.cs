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
            private static int allocatedMemory = 0;

            public MemoryAllocation(IMem buffer, int size, MemFlags flags)
            {
                this.buffer = buffer;
                this.bufferSizeInBytes = size;
                this.flags = flags;
                allocatedMemory += bufferSizeInBytes;
            }

            ~MemoryAllocation()
            {
                Release();
            }

            internal void Release()
            {
                if (buffer != null)
                {
                    allocatedMemory -= bufferSizeInBytes;
                    Cl.ReleaseMemObject(buffer);
                    buffer = null;
                }
            }
            internal static int GetUsedMemory()
            {
                return allocatedMemory;
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

        public MemoryAllocation CreateMemoryAllocation(Context clContext, int byteSize, MemFlags flags, IntPtr data)
        {
            ErrorCode err;
            var buffer = Cl.CreateBuffer(clContext, flags, new IntPtr(byteSize), data, out err);
            if (err != ErrorCode.Success)
            {
                OnLowDeviceMemory();
                buffer = Cl.CreateBuffer(clContext, flags, new IntPtr(byteSize), data, out err);
                if (err != ErrorCode.Success)
                    ThrowOnError(err, String.Format("Failed to allocate device memory. Size: {0}", byteSize));
            }
            return new MemoryAllocation(buffer, byteSize, flags);
        }

        private void OnLowDeviceMemory()
        {
            foreach (var item in freeMemoryAllocations)
                item.Release();
            freeMemoryAllocations.Clear();
        }

        public void FlushWorkingCache()
        {
            foreach (var item in freeMemoryAllocations)
                item.Release();
            freeMemoryAllocations.Clear();

            //usedMemoryAllocations should be empty, but delete its content just to be safe
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

        public void UploadToMemory(MemoryAllocation mem, int offsetInBytes, int sizeInBytes, IntPtr data, bool IsBlocking)
        {
            Event e;
            var errCodeWrite = Cl.EnqueueWriteBuffer(commandQueue, mem.buffer, IsBlocking ? Bool.True : Bool.False, new IntPtr(offsetInBytes), new IntPtr(sizeInBytes), data, 0, null, out e);
            ThrowOnError(errCodeWrite, String.Format("Failed to enqueue write buffer. Write-size:{0}, Target buffer size: {1}", sizeInBytes, mem.bufferSizeInBytes));

            var errCodeEv = Cl.ReleaseEvent(e);
            ThrowOnError(errCodeEv, String.Format("Failed release event (EnqueueWriteBuffer, UploadToMemory_1)"));
        }

        public void UploadToMemory(MemoryAllocation mem, int offset, int[] data, bool IsBlocking, int size = -1)
        {
            int uploadSize = size < 0 ? (data.Length * 4) : size * 4;
            Event e;
            ErrorCode errCodeWrite;
            unsafe
            {
                fixed (int* dataPtr = data)
                {
                    errCodeWrite = Cl.EnqueueWriteBuffer(commandQueue, mem.buffer, IsBlocking ? Bool.True : Bool.False, IntPtr.Zero, new IntPtr(uploadSize), new IntPtr(dataPtr + offset), 0, null, out e);
                }
            }
            ThrowOnError(errCodeWrite, String.Format("Failed to enqueue write buffer. Write-size:{0}, Target buffer size: {1}, offset:{2}", uploadSize, mem.bufferSizeInBytes, offset*4));

            var errCodeEv = Cl.ReleaseEvent(e);
            ThrowOnError(errCodeEv, String.Format("Failed release event (EnqueueWriteBuffer, UploadToMemory_2)"));
        }

        public void UploadToMemory(MemoryAllocation mem, int offset, float[] data, bool IsBlocking, int size = -1)
        {
            int uploadSize = size < 0 ? (data.Length * 4) : size * 4;
            Event e;
            ErrorCode errCodeWrite;
            unsafe
            {
                fixed (float* dataPtr = data)
                {
                    errCodeWrite = Cl.EnqueueWriteBuffer(commandQueue, mem.buffer, IsBlocking ? Bool.True : Bool.False, IntPtr.Zero, new IntPtr(uploadSize), new IntPtr(dataPtr + offset), 0, null, out e);
                }
            }
            ThrowOnError(errCodeWrite, String.Format("Failed to enqueue write buffer. Write-size:{0}, Target buffer size: {1}, offset:{2}", uploadSize, mem.bufferSizeInBytes, offset * 4));

            var errCodeEv = Cl.ReleaseEvent(e);
            ThrowOnError(errCodeEv, String.Format("Failed release event (EnqueueWriteBuffer, UploadToMemory_3)"));
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

        public MemoryAllocation GetMemoryFor(int requiredSizeInBytes, MemFlags flags, IntPtr data )
        {
            if (!IsInitialized())
                return null;

            var accessFlags = flags & (MemFlags.ReadOnly | MemFlags.WriteOnly | MemFlags.ReadWrite);

            MemoryAllocation candidate = null;
            int bestMatchingSize = int.MaxValue;
            int itemToSwap = -1;

            for (int i = 0; i < freeMemoryAllocations.Count; i++)
            {
                var item = freeMemoryAllocations[i];
                if (item.flags == accessFlags && item.bufferSizeInBytes >= requiredSizeInBytes && item.bufferSizeInBytes < bestMatchingSize) //Select the smallest sufficient memory allocation from our allocations
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
                candidate = CreateMemoryAllocation(clContext, requiredSizeInBytes, accessFlags, IntPtr.Zero);
                usedMemoryAllocations.Add(candidate);
            }
            else
            {
                freeMemoryAllocations.RemoveAt(itemToSwap);
                usedMemoryAllocations.Add(candidate);
            }

            if (flags.HasFlag(MemFlags.CopyHostPtr) && data != IntPtr.Zero)
            {
                Event e;
                var errCodeWrite = Cl.EnqueueWriteBuffer(commandQueue, candidate.buffer, Bool.False, IntPtr.Zero, new IntPtr(requiredSizeInBytes), data, 0, null, out e);
                ThrowOnError(errCodeWrite, String.Format("Failed to enqueue write buffer. Write-size:{0}, Target buffer size: {1}", requiredSizeInBytes, candidate.bufferSizeInBytes));

                var errCodeEv = Cl.ReleaseEvent(e);
                ThrowOnError(errCodeEv, String.Format("Failed release event (EnqueueWriteBuffer)"));
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

        private void ThrowOnError(ErrorCode err, string message)
        {
            if ( err != ErrorCode.Success)
                throw new Exception(message + " Error code: " + err.ToString());
        }

        public void SetKernelArg(string kernelName, uint idx, MemoryAllocation mem)
        {
            var errCode = Cl.SetKernelArg(kernels[kernelName], idx, mem.buffer);
            ThrowOnError(errCode, String.Format("Failed to set arg #{0} for kernel {1}", idx, kernelName));
        }

        public void ReadBuffer(MemoryAllocation mem, bool isBlocking, IntPtr offset, IntPtr lengthInBytes, IntPtr output)
        {
            Event ev;
            var errCodeRead = Cl.EnqueueReadBuffer(commandQueue, mem.buffer, isBlocking ? Bool.True : Bool.False, offset, lengthInBytes, output, 0, null, out ev);
            ThrowOnError(errCodeRead, String.Format("Failed to enqueue read buffer. Read Size: {0}, Buffer size: {1}", lengthInBytes, mem.bufferSizeInBytes));

            var errCodeEv = Cl.ReleaseEvent(ev);
            ThrowOnError(errCodeEv, String.Format("Failed release event (EnqueueReadBuffer)"));
        }

        public void BlockUntilAllTasksDone()
        {
            var errCode = Cl.Finish(commandQueue);
            ThrowOnError(errCode, String.Format("Failed clFinish() call"));
        }

        public void EnqueueKernel(string kernelName, IntPtr[] globalWorkSize, IntPtr[] localWorkSize)
        {
            Event ev;
            var errCodeRunKernel = Cl.EnqueueNDRangeKernel(commandQueue, kernels[kernelName], (uint)globalWorkSize.Length, null, globalWorkSize, localWorkSize, 0, null, out ev);
            ThrowOnError(errCodeRunKernel, String.Format("Failed to enqueue kernel {0}.", kernelName));

            var errCodeEv = Cl.ReleaseEvent(ev);
            ThrowOnError(errCodeEv, String.Format("Failed release event (EnqueueNDRangeKernel)"));
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

        public int GetUsedMemory() { return MemoryAllocation.GetUsedMemory(); }

        internal void FlushCommandBuffer()
        {
            Cl.Flush(commandQueue);
        }
    }
}
