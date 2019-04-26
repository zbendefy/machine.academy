using Mademy;
using Mademy.OpenCL;
using OpenCL.Net;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Mademy.OpenCL.ComputeFramework;

namespace Mademy
{
    public class MathLib
    {
        private ComputeFramework computeFramework = null;
        private static readonly string calcLayerKernel = "calcLayer";

        public MathLib(ComputeDevice clDevice = null)
        {
            if ( clDevice != null)
                computeFramework = new ComputeFramework(clDevice, new string[] { CLSourceProvider.ReadSourceFile() }, new string[] { calcLayerKernel } , "-cl-finite-math-only -Werror");
        }

        private bool HasComputeFramework() { return computeFramework != null; }

        public MathLib Clone()
        {
            return new MathLib(HasComputeFramework() ? computeFramework.GetOpenCLDevice() : null);
        }

        internal float[] CalculateLayer(float[,] weightMx, float[] bias, float[] prevActivations, IActivationFunction sigmoidFunction)
        {
            if (!HasComputeFramework()) //CPU fallback
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
                    mem_param_weightMx = computeFramework.GetMemoryFor(weightMx.Length * 4, MemFlags.ReadOnly | MemFlags.CopyHostPtr, new IntPtr(weightArrayPtr));
                }
                fixed (float* biasPtr = bias)
                {
                    mem_param_bias = computeFramework.GetMemoryFor(bias.Length * 4, MemFlags.ReadOnly | MemFlags.CopyHostPtr, new IntPtr(biasPtr));
                }
                fixed (float* prevActivationPtr = prevActivations)
                {
                    mem_param_prevActivation = computeFramework.GetMemoryFor(prevActivations.Length * 4, MemFlags.ReadOnly | MemFlags.CopyHostPtr, new IntPtr(prevActivationPtr));
                }

                int[] configParams = new int[] { /*rows: */weightMx.GetLength(0), /*cols: */weightMx.GetLength(1), /*ApplySigmoid*/ sigmoidFunction.GetOpenCLFunctionId() };
                fixed (int* configPtr = configParams)
                {
                    mem_param_config = computeFramework.GetMemoryFor(configParams.Length * 4, MemFlags.ReadOnly | MemFlags.CopyHostPtr, new IntPtr(configPtr));
                }
                mem_param_output = computeFramework.GetMemoryFor(matrixRows * 4, MemFlags.WriteOnly, IntPtr.Zero);
            }

            computeFramework.SetKernelArg(calcLayerKernel, 0, mem_param_weightMx);
            computeFramework.SetKernelArg(calcLayerKernel, 1, mem_param_bias);
            computeFramework.SetKernelArg(calcLayerKernel, 2, mem_param_prevActivation);
            computeFramework.SetKernelArg(calcLayerKernel, 3, mem_param_config);
            computeFramework.SetKernelArg(calcLayerKernel, 4, mem_param_output);

            int localWorkgroupSize = 32;
            int globalWorkSize = (matrixRows % localWorkgroupSize == 0) ? matrixRows : (matrixRows + (localWorkgroupSize - (matrixRows % localWorkgroupSize)));
            computeFramework.EnqueueKernel(calcLayerKernel, new IntPtr[] { new IntPtr(globalWorkSize) }, new IntPtr[] { new IntPtr(localWorkgroupSize) });

            float[] output = new float[matrixRows];

            unsafe
            {
                fixed (float* outputPtr = output)
                {
                    computeFramework.ReadBuffer(mem_param_output, true, IntPtr.Zero, new IntPtr(matrixRows * 4), new IntPtr(outputPtr));
                }
            }

            computeFramework.UnuseMemoryAllocations();

            return output;
        }

        private void CalculateGradientForSingleTrainingExample( Network network, IErrorFunction errorFunction, ref List<List<NeuronData>> intermediateResults, float[] trainingInput, float[] trainingDesiredOutput)
        {
            List<float[]> activations = new List<float[]>();
            List<float[]> zValues = new List<float[]>();
            network.Compute(this, trainingInput, ref activations, ref zValues, false); //dont flush working cache

            var lastLayerGradient = intermediateResults.Last();
            List<float> delta_k_holder = new List<float>();
            CalculateOutputLayerGradient(network, errorFunction, ref lastLayerGradient, ref delta_k_holder, activations, trainingInput, zValues, trainingDesiredOutput);

            for (int i = network.layers.Count - 2; i >= 0; --i)
            {
                var layerGradient = intermediateResults[i];
                CalculateHiddenLayerGradient(network, i, ref layerGradient, ref delta_k_holder, i == 0 ? trainingInput : activations[i - 1], zValues);
            }
        }

        private void CalculateOutputLayerGradient(Network network, IErrorFunction errorFunction, ref List<NeuronData> gradientData, ref List<float> gamma_k_vector, List<float[]> activations, float[] trainingInput, List<float[]> zValues, float[] desiredOutput)
        {
            var prevActivations = activations.Count <= 1 ? trainingInput : activations[activations.Count - 2];
            int lastLayerWeightCount = network.layers.Last().GetWeightsPerNeuron();
            int lastLayerNeuronCount = network.layers.Last().GetNeuronCount();
            for (int i = 0; i < lastLayerNeuronCount; i++)
            {
                float outputValue = activations.Last()[i];
                float gamma_k = errorFunction.CalculateDelta(zValues.Last()[i], outputValue, desiredOutput[i], network.activationFunction);

                var gradientDataItem = gradientData[i];
                //Assert(gradientData[i].weights.Length == prevActivations.Length);
                for (int j = 0; j < lastLayerWeightCount; j++)
                {
                    gradientDataItem.weights[j] += gamma_k * (prevActivations[j]);
                }
                gradientDataItem.bias += gamma_k;
                gamma_k_vector.Add(gamma_k);
            }
        }
        private void CalculateHiddenLayerGradient(Network network, int L, ref List<NeuronData> gradientData, ref List<float> gamma_k_vector, float[] prevLayerActivations, List<float[]> zValues)
        {
            List<float> newGammak = new List<float>();
            int layerWeightCount = network.layers[L].GetWeightsPerNeuron();
            int layerNeuronCount = network.layers[L].GetNeuronCount();

            for (int i = 0; i < layerNeuronCount; i++)
            {
                float gamma_j = 0;
                //Assert(gamma_k_vector.Count == layers[L + 1].weightMx.GetLength(0));
                for (int k = 0; k < gamma_k_vector.Count; k++)
                {
                    gamma_j += gamma_k_vector[k] * network.layers[L + 1].weightMx[k, i];
                }
                gamma_j *= network.activationFunction.CalculatePrime(zValues[L][i]);
                newGammak.Add(gamma_j);

                //Assert(gradientData[i].weights.Length == prevLayerActivations.Length);
                var gradientDataItem = gradientData[i];
                for (int j = 0; j < layerWeightCount; j++)
                {
                    gradientDataItem.weights[j] += gamma_j * (prevLayerActivations[j]);
                }
                gradientDataItem.bias += gamma_j;
            }

            gamma_k_vector = newGammak;
        }

        internal void FlushWorkingCache()
        {
            if ( HasComputeFramework())
                computeFramework.FlushWorkingCache();
        }

        /// <summary>
        /// Runs backpropagation
        /// </summary>
        /// <param name="network"></param>
        /// <param name="suite"></param>
        /// <param name="trainingDataBegin"></param>
        /// <param name="trainingDataEnd"></param>
        /// <returns></returns>
        internal List<List<NeuronData>> CalculateAccumulatedGradientForMinibatch(Network network, TrainingSuite suite, int trainingDataBegin, int trainingDataEnd)
        {
            //Backpropagation
            var ret = Utils.CreateGradientVector(network);

            if (!HasComputeFramework() || true) //CPU fallback
            {
                for (int i = trainingDataBegin; i < trainingDataEnd; i++)
                {
                    CalculateGradientForSingleTrainingExample(network, suite.config.costFunction, ref ret, suite.trainingData[i].input, suite.trainingData[i].desiredOutput);
                }
            }
            
            //TODO run whole minibatch on the OpenCL device

            return ret;
        }

    }
}
