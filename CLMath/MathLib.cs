using OpenCL.Net;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CLMath
{
    public class MathLib
    {
        private static string calcLayerKernel = "calcLayer";

        Context clContext;
        ComputeDevice clDevice = null;
        CommandQueue commandQueue;
        Program clProgram;
        Dictionary<String, Kernel> kernels = new Dictionary<string, Kernel>();
        bool hasClInitialized = false;

        public MathLib(ComputeDevice clDevice = null)
        {
            this.clDevice = clDevice;
            InitCL();
        }

        public static float Sigmoid(float x)
        {
            return 1.0f / (1.0f + (float)Math.Exp(-x));
        }

        public static float SigmoidPrime(float x)
        {
            return Sigmoid(x) * (1.0f - Sigmoid(x));
        }

        private void CleanupCLResources()
        {
            if (hasClInitialized)
            {
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

        private void InitCL()
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

        public float[] CalculateLayer(float[,] weightMx, float[] bias, float[] prevActivations)
        {
            return CalculateLayer(weightMx, bias, prevActivations, true);
        }

        private float[] CalculateLayer(float[,] weightMx, float[] bias, float[] prevActivations, bool applySigmoid)
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
                    if ( applySigmoid )
                        acc = Sigmoid(acc);
                    ret[m] = acc;
                }
                return ret;
            }

            int matrixRows = weightMx.GetLength(0);

            int[] configParams = new int[] { /*rows: */weightMx.GetLength(0), /*cols: */weightMx.GetLength(1), /*ApplySigmoid*/ applySigmoid ? 1 : 0 };
            float[] output = new float[matrixRows];

            float[] weightMxContinous = new float[weightMx.GetLength(0) * weightMx.GetLength(1)];
            Buffer.BlockCopy(weightMx, 0, weightMxContinous, 0, weightMx.GetLength(0) * weightMx.GetLength(1) * 4);

            ErrorCode err;
            var mem_param_weightMx = Cl.CreateBuffer<float>(clContext, MemFlags.ReadOnly | MemFlags.CopyHostPtr, weightMxContinous, out err);
            var mem_param_bias = Cl.CreateBuffer<float>(clContext, MemFlags.ReadOnly | MemFlags.CopyHostPtr, bias, out err);
            var mem_param_prevActivation = Cl.CreateBuffer<float>(clContext, MemFlags.ReadOnly | MemFlags.CopyHostPtr, prevActivations, out err);
            var mem_param_config = Cl.CreateBuffer<int>(clContext, MemFlags.ReadOnly | MemFlags.CopyHostPtr, configParams, out err);
            var mem_param_output = Cl.CreateBuffer<float>(clContext, MemFlags.WriteOnly, output.Length, out err);

            var kernel = kernels[calcLayerKernel];
            Cl.SetKernelArg(kernel, 0, mem_param_weightMx);
            Cl.SetKernelArg(kernel, 1, mem_param_bias);
            Cl.SetKernelArg(kernel, 2, mem_param_prevActivation);
            Cl.SetKernelArg(kernel, 3, mem_param_config);
            Cl.SetKernelArg(kernel, 4, mem_param_output);

            Event ev;
            int localWorkgroupSize = 32;
            int globalWorkSize = (matrixRows % localWorkgroupSize == 0) ? matrixRows : (matrixRows + (localWorkgroupSize - (matrixRows % localWorkgroupSize)));
            Cl.EnqueueNDRangeKernel(commandQueue, kernel, 1, null, new IntPtr[] { new IntPtr(globalWorkSize) }, new IntPtr[] { new IntPtr(localWorkgroupSize) }, 0, null, out ev);
            Cl.EnqueueReadBuffer(commandQueue, mem_param_output, Bool.True, 0, matrixRows, output, 0, null, out ev);

            Cl.ReleaseMemObject(mem_param_weightMx);
            Cl.ReleaseMemObject(mem_param_bias);
            Cl.ReleaseMemObject(mem_param_prevActivation);
            Cl.ReleaseMemObject(mem_param_config);
            Cl.ReleaseMemObject(mem_param_output);

            return output;
        }

        public float[] CalculateZ(float[,] weightMx, float[] bias, float[] prevActivations)
        {
            return CalculateLayer(weightMx, bias, prevActivations, false);
        }
        
    }
}
