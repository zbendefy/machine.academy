using OpenCl.DotNetCore.Interop;
using OpenCl.DotNetCore.Interop.CommandQueues;
using OpenCl.DotNetCore.Interop.Contexts;
using OpenCl.DotNetCore.Interop.EnqueuedCommands;
using OpenCl.DotNetCore.Interop.Events;
using OpenCl.DotNetCore.Interop.Kernels;
using OpenCl.DotNetCore.Interop.Memory;
using OpenCl.DotNetCore.Interop.Programs;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace Macademy.OpenCL
{
    internal class ComputeFramework
    {
        public class MemoryAllocation
        {
            public IntPtr buffer = IntPtr.Zero;
            public uint bufferSizeInBytes;
            public MemoryFlag flags;
            private static uint allocatedMemory = 0;

            public UIntPtr GetAllocationSize(){return new UIntPtr(bufferSizeInBytes);}

            public MemoryAllocation(IntPtr buffer, uint size, MemoryFlag flags)
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
                if (buffer != IntPtr.Zero)
                {
                    allocatedMemory -= bufferSizeInBytes;
                    MemoryNativeApi.ReleaseMemoryObject(buffer);
                    buffer = IntPtr.Zero;
                }
            }
            internal static int GetUsedMemory()
            {
                return (int)allocatedMemory;
            }
        }

        private object lockObj = new object();
        private IntPtr clContext;
        private IntPtr clDevice;
        private IntPtr commandQueue;
        private IntPtr clProgram;
        private Dictionary<String, IntPtr> kernels = new Dictionary<string, IntPtr>();
        private bool hasClInitialized = false;
        private List<MemoryAllocation> freeMemoryAllocations = new List<MemoryAllocation>();
        private List<MemoryAllocation> usedMemoryAllocations = new List<MemoryAllocation>();

        public ComputeFramework(IntPtr clDevice, String[] kernelSource, String[] kernelNames, string compileArguments)
        {
            this.clDevice = clDevice;
            InitCL(kernelSource, kernelNames, compileArguments);
        }

        ~ComputeFramework() { CleanupCLResources(); }

        public MemoryAllocation CreateMemoryAllocation(IntPtr clContext, uint byteSize, MemoryFlag flags, IntPtr data)
        {
            Result err;
            var buffer = MemoryNativeApi.CreateBuffer(clContext, flags, new UIntPtr(byteSize), data, out err);
            if (err != Result.Success)
            {
                OnLowDeviceMemory();
                buffer = MemoryNativeApi.CreateBuffer(clContext, flags, new UIntPtr(byteSize), data, out err);
                if (err != Result.Success)
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
                    KernelsNativeApi.ReleaseKernel(item.Value);
                }
                kernels.Clear();

                ProgramsNativeApi.ReleaseProgram(clProgram);
                CommandQueuesNativeApi.ReleaseCommandQueue(commandQueue);
                ContextsNativeApi.ReleaseContext(clContext);

                hasClInitialized = false;
            }
        }

        public bool IsInitialized() { return hasClInitialized; }

        public void UploadToMemory(MemoryAllocation mem, int mem_offset, int sizeInBytes, IntPtr data, bool IsBlocking)
        {
            IntPtr ev = IntPtr.Zero;
            var errCodeWrite = EnqueuedCommandsNativeApi.EnqueueWriteBuffer(commandQueue, mem.buffer, IsBlocking ? 1U : 0U, new UIntPtr((uint)mem_offset*4U), new UIntPtr((uint)sizeInBytes), data, 0, null, out ev);
            ThrowOnError(errCodeWrite, String.Format("Failed to enqueue write buffer. Write-size:{0}, Target buffer size: {1}", sizeInBytes, mem.bufferSizeInBytes));

            var errCodeEv = EventsNativeApi.ReleaseEvent(ev);
            ThrowOnError(errCodeEv, String.Format("Failed release event (EnqueueWriteBuffer, UploadToMemory_1)"));
        }

        public void UploadToMemory(MemoryAllocation mem, int mem_offset, int data_offset, int[] data, bool IsBlocking, int size = -1)
        {
            uint uploadSize = size < 0 ? ((uint)data.Length * 4U) : (uint)size * 4U;
            IntPtr ev = IntPtr.Zero;
            Result errCodeWrite;
            unsafe
            {
                fixed (int* dataPtr = data)
                {
                    errCodeWrite = EnqueuedCommandsNativeApi.EnqueueWriteBuffer(commandQueue, mem.buffer, IsBlocking ? 1U : 0U, new UIntPtr((uint)mem_offset*4U), new UIntPtr(uploadSize), new IntPtr(dataPtr + data_offset), 0, null, out ev);
                }
            }
            ThrowOnError(errCodeWrite, String.Format("Failed to enqueue write buffer. Write-size:{0}, Target buffer size: {1}, data_offset:{2}", uploadSize, mem.bufferSizeInBytes, data_offset*4));

            var errCodeEv = EventsNativeApi.ReleaseEvent(ev);
            ThrowOnError(errCodeEv, String.Format("Failed release event (EnqueueWriteBuffer, UploadToMemory_2)"));
        }

        public void UploadToMemory(MemoryAllocation mem, int mem_offset, int data_offset, float[] data, bool IsBlocking, int size = -1)
        {
            uint uploadSize = size < 0 ? ((uint)data.Length * 4U) : (uint)size * 4U;
            IntPtr ev = IntPtr.Zero;
            Result errCodeWrite;
            unsafe
            {
                fixed (float* dataPtr = data)
                {
                    errCodeWrite = EnqueuedCommandsNativeApi.EnqueueWriteBuffer(commandQueue, mem.buffer, IsBlocking ? 1U : 0U, new UIntPtr((uint)mem_offset), new UIntPtr(uploadSize), new IntPtr(dataPtr + data_offset), 0, null, out ev);
                }
            }
            ThrowOnError(errCodeWrite, String.Format("Failed to enqueue write buffer. Write-size:{0}, Target buffer size: {1}, data_offset:{2}", uploadSize, mem.bufferSizeInBytes, data_offset * 4));

            var errCodeEv = EventsNativeApi.ReleaseEvent(ev);
            ThrowOnError(errCodeEv, String.Format("Failed to release event (EnqueueWriteBuffer, UploadToMemory_3)"));
        }

        public MemoryAllocation GetMemoryFor(int requiredSizeInBytes, MemoryFlag flags, IntPtr data, int data_size_in_bytes = -1)
        {
            if (!IsInitialized())
                return null;

            var accessFlags = flags & (MemoryFlag.ReadOnly | MemoryFlag.WriteOnly | MemoryFlag.ReadWrite);

            MemoryAllocation candidate = null;
            uint bestMatchingSize = uint.MaxValue;
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
                candidate = CreateMemoryAllocation(clContext, (uint)requiredSizeInBytes, accessFlags, IntPtr.Zero);
                usedMemoryAllocations.Add(candidate);
            }
            else
            {
                freeMemoryAllocations.RemoveAt(itemToSwap);
                usedMemoryAllocations.Add(candidate);
            }

            if (flags.HasFlag(MemoryFlag.CopyHostPointer) && data != IntPtr.Zero)
            {
                int upload_size_in_bytes = data_size_in_bytes >= 0 ? data_size_in_bytes : requiredSizeInBytes;

                IntPtr ev = IntPtr.Zero;
                var errCodeWrite = EnqueuedCommandsNativeApi.EnqueueWriteBuffer(commandQueue, candidate.buffer, 0U, UIntPtr.Zero, new UIntPtr((uint)upload_size_in_bytes), data, 0, null, out ev);
                ThrowOnError(errCodeWrite, String.Format("Failed to enqueue write buffer. Write-size:{0}, Target buffer size: {1}", requiredSizeInBytes, candidate.bufferSizeInBytes));

                var errCodeEv = EventsNativeApi.ReleaseEvent(ev);
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

        private void ThrowOnError(Result err, string message)
        {
            if ( err != Result.Success)
                throw new OpenClException(message, err);
        }

        public void SetKernelArg(string kernelName, uint idx, MemoryAllocation mem)
        {
            GCHandle garbageCollectorHandle = GCHandle.Alloc(mem.buffer, GCHandleType.Pinned);
            var errCode = KernelsNativeApi.SetKernelArgument(kernels[kernelName], idx, new UIntPtr((uint)Marshal.SizeOf(mem.buffer)), garbageCollectorHandle.AddrOfPinnedObject());
            garbageCollectorHandle.Free();
            ThrowOnError(errCode, String.Format("Failed to set arg #{0} for kernel {1}", idx, kernelName));
        }

        public void ReadBuffer(MemoryAllocation mem, bool isBlocking, UIntPtr offset, UIntPtr lengthInBytes, IntPtr output)
        {
            IntPtr ev = IntPtr.Zero;
            var errCodeRead = EnqueuedCommandsNativeApi.EnqueueReadBuffer(commandQueue, mem.buffer, isBlocking ? 1U : 0U, offset, lengthInBytes, output, 0, null, out ev);
            ThrowOnError(errCodeRead, String.Format("Failed to enqueue read buffer. Read Size: {0}, Buffer size: {1}", lengthInBytes, mem.bufferSizeInBytes));

            var errCodeEv = EventsNativeApi.ReleaseEvent(ev);
            ThrowOnError(errCodeEv, String.Format("Failed release event (EnqueueReadBuffer)"));
        }

        public void BlockUntilAllTasksDone()
        {
            var errCode = CommandQueuesNativeApi.Finish(commandQueue);
            ThrowOnError(errCode, String.Format("Failed clFinish() call"));
        }

        public void EnqueueKernel(string kernelName, IntPtr[] globalWorkSize, IntPtr[] localWorkSize)
        {
            IntPtr ev = IntPtr.Zero;
            var errCodeRunKernel = EnqueuedCommandsNativeApi.EnqueueNDRangeKernel(commandQueue, kernels[kernelName], (uint)globalWorkSize.Length, null, globalWorkSize, localWorkSize, 0, null, out ev);
            ThrowOnError(errCodeRunKernel, String.Format("Failed to enqueue kernel {0}.", kernelName));

            var errCodeEv = EventsNativeApi.ReleaseEvent(ev);
            ThrowOnError(errCodeEv, String.Format("Failed release event (EnqueueNDRangeKernel)"));
        }


        /// <summary>
        /// Retrieves the specified information about the program build.
        /// </summary>
        /// <typeparam name="T">The type of the data that is to be returned.</param>
        /// <param name="program">The handle to the program for which the build information is to be retrieved.</param>
        /// <param name="device">The device for which the build information is to be retrieved.</param>
        /// <param name="programBuildInformation">The kind of information that is to be retrieved.</param>
        /// <exception cref="OpenClException">If the information could not be retrieved, then an <see cref="OpenClException"/> is thrown.</exception>
        /// <returns>Returns the specified information.</returns>
        private T GetProgramBuildInformation<T>(IntPtr program, IntPtr device, ProgramBuildInformation programBuildInformation)
        {
            // Retrieves the size of the return value in bytes, this is used to later get the full information
            UIntPtr returnValueSize;
            Result result = ProgramsNativeApi.GetProgramBuildInformation(program, device, programBuildInformation, UIntPtr.Zero, null, out returnValueSize);
            if (result != Result.Success)
                throw new OpenClException("The program build information could not be retrieved.", result);
            
            // Allocates enough memory for the return value and retrieves it
            byte[] output = new byte[returnValueSize.ToUInt32()];
            result = ProgramsNativeApi.GetProgramBuildInformation(program, device, programBuildInformation, new UIntPtr((uint)output.Length), output, out returnValueSize);
            if (result != Result.Success)
                throw new OpenClException("The program build information could not be retrieved.", result);

            // Returns the output
            return InteropConverter.To<T>(output);
        }

        private void InitCL(String[] kernelSource, String[] kernelNames, string compileArguments)
        {
            lock (lockObj)
            {
                if (!hasClInitialized && clDevice != null)
                {
                    Result err;
                    var devicesArray = new IntPtr[] { clDevice };
                    clContext = ContextsNativeApi.CreateContext(IntPtr.Zero, 1, devicesArray, IntPtr.Zero, IntPtr.Zero, out err);
                    if (err != Result.Success) throw new OpenClException("Failed to create context!", err);

                    commandQueue = CommandQueuesNativeApi.CreateCommandQueue(clContext, clDevice, CommandQueueProperty.None, out err);
                    if (err != Result.Success) throw new OpenClException("Failed to create command queue!", err);

                    IntPtr[] sourceList = kernelSource.Select(source => Marshal.StringToHGlobalAnsi(source)).ToArray();
                    clProgram =  ProgramsNativeApi.CreateProgramWithSource(clContext, 1, sourceList, null, out err);
                    if (err != Result.Success) throw new OpenClException("Failed to create program!", err);

                    err = ProgramsNativeApi.BuildProgram(clProgram, 1, new IntPtr[] { clDevice }, compileArguments, IntPtr.Zero, IntPtr.Zero);
                    if (err != Result.Success)
                    {
                        var infoBuffer = GetProgramBuildInformation<string>(clProgram, clDevice, ProgramBuildInformation.Log);
                        if (err != Result.Success) throw new OpenClException("Failed to build program! " + (infoBuffer == null ? "?" : infoBuffer.ToString()), err);
                    }

                    foreach (var item in kernelNames)
                    {
                        kernels[item] = KernelsNativeApi.CreateKernel(clProgram, item, out err);
                        if (err != Result.Success) throw new OpenClException("Failed to create kernel: " + item, err);
                    }

                    hasClInitialized = true;
                }
            }
        }

        public int GetUsedMemory() { return MemoryAllocation.GetUsedMemory(); }

        internal void FlushCommandBuffer()
        {
            CommandQueuesNativeApi.Flush(commandQueue);
        }
    }
}
