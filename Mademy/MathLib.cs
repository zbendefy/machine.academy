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
        private static readonly string calcLayerKernel = "calcSingleLayer";
        private static readonly string forwardPass = "trainingForwardPass";
        private static readonly string backwardPassOutputlayer = "trainingOutputLayer";
        private static readonly string backwardPassHiddenlayer = "trainingHiddenLayer";

        public MathLib(ComputeDevice clDevice = null)
        {
            if ( clDevice != null)
                computeFramework = new ComputeFramework(clDevice, new string[] { CLSourceProvider.ReadSourceFile() }, new string[] { calcLayerKernel } , "-cl-finite-math-only -Werror");
        }

        /// <summary>
        /// Extends the global work size to the nearest upper multiple of the localSize
        /// </summary>
        /// <param name="desiredGlobalSize"></param>
        /// <param name="localSize"></param>
        /// <returns></returns>
        private int ExtendGlobalWorkSize(int desiredGlobalSize, int localSize)
        {
            return ((desiredGlobalSize % localSize) == 0) ? desiredGlobalSize : (desiredGlobalSize + (localSize - (desiredGlobalSize % localSize)));
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
            int globalWorkSize = ExtendGlobalWorkSize(matrixRows, localWorkgroupSize);
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

        private void CalculateOutputLayerGradient(Network network, IErrorFunction errorFunction, ref List<NeuronData> gradientData, ref List<float> delta_k_vector, List<float[]> activations, float[] trainingInput, List<float[]> zValues, float[] desiredOutput)
        {
            var prevActivations = activations.Count <= 1 ? trainingInput : activations[activations.Count - 2];
            int lastLayerWeightCount = network.layers.Last().GetWeightsPerNeuron();
            int lastLayerNeuronCount = network.layers.Last().GetNeuronCount();
            for (int i = 0; i < lastLayerNeuronCount; i++)
            {
                float outputValue = activations.Last()[i];
                float delta_k = errorFunction.CalculateDelta(zValues.Last()[i], outputValue, desiredOutput[i], network.activationFunction);

                var gradientDataItem = gradientData[i];
                //Assert(gradientData[i].weights.Length == prevActivations.Length);
                for (int j = 0; j < lastLayerWeightCount; j++)
                {
                    gradientDataItem.weights[j] += delta_k * (prevActivations[j]);
                }
                gradientDataItem.bias += delta_k;
                delta_k_vector.Add(delta_k);
            }
        }
        private void CalculateHiddenLayerGradient(Network network, int L, ref List<NeuronData> gradientData, ref List<float> delta_k_vector, float[] prevLayerActivations, List<float[]> zValues)
        {
            List<float> newGammak = new List<float>();
            int layerWeightCount = network.layers[L].GetWeightsPerNeuron();
            int layerNeuronCount = network.layers[L].GetNeuronCount();

            for (int i = 0; i < layerNeuronCount; i++)
            {
                float gamma_j = 0;
                //Assert(delta_k_vector.Count == layers[L + 1].weightMx.GetLength(0));
                for (int k = 0; k < delta_k_vector.Count; k++)
                {
                    gamma_j += delta_k_vector[k] * network.layers[L + 1].weightMx[k, i];
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

            delta_k_vector = newGammak;
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
            int trainingSamples = trainingDataEnd - trainingDataBegin;


            int[] networkConfigParams = null;
            int totalWeightAndBiasCount = 0;
            int widestLayerNeuronCount = 0;
            {
                List<int> networkConfigParamsList = new List<int>();
                networkConfigParamsList.Add(0); //layer index to be processed
                networkConfigParamsList.Add(network.layers.Count); //Layer count
                networkConfigParamsList.Add(trainingSamples); //Layer count
                networkConfigParamsList.Add(network.activationFunction.GetOpenCLFunctionId()); //Activation function
                networkConfigParamsList.Add(suite.config.costFunction.GetOpenCLFunctionID()); //Cost function
                networkConfigParamsList.Add(network.layers.First().GetWeightsPerNeuron()); //Input count
                for (int i = 0; i < network.layers.Count; i++)
                {
                    networkConfigParamsList.Add(network.layers[i].GetNeuronCount()); //Layer neuron count
                    totalWeightAndBiasCount += network.layers[i].biases.Length;
                    totalWeightAndBiasCount += network.layers[i].weightMx.Length;
                    widestLayerNeuronCount = Math.Max(network.layers[i].GetNeuronCount(), widestLayerNeuronCount);
                }
                networkConfigParams = networkConfigParamsList.ToArray();
            }
            MemoryAllocation mem_NetworkConfigParams = computeFramework.GetMemoryFor( MemFlags.ReadOnly | MemFlags.CopyHostPtr, networkConfigParams );

            int inputActivationCount = network.layers.First().GetWeightsPerNeuron();
            float[] inputParameters = new float[trainingSamples * inputActivationCount];
            for (int i = trainingDataBegin; i < trainingDataEnd; ++i)
                Buffer.BlockCopy(suite.trainingData[i].input, 0, inputParameters, i * inputActivationCount * 4, inputActivationCount);
            MemoryAllocation mem_InputActivations = computeFramework.GetMemoryFor(MemFlags.ReadOnly, inputParameters);

            int totalActivationCount = 0; //Add 
            foreach (var item in network.layers)
            {
                totalActivationCount += item.GetNeuronCount();
            }
            totalActivationCount *= trainingSamples;
            ///Contains the whole network's activation values, and Z values for each training sample
            ///Memory layout for one layer is like this: [...input values...][...first layer's activations...][...second layer's activations]...[last layer's activations][first layer's z values][second layer's zvalues]...[last layer's z values]
            ///After that, the next layer's same values are there
            MemoryAllocation mem_activationsAndZValues = computeFramework.GetMemoryFor(totalActivationCount * 2 * 4, MemFlags.ReadWrite, IntPtr.Zero);


            float[] weightsAndBiases = new float[totalWeightAndBiasCount];
            {
                int offset = 0;
                foreach (var layer in network.layers)
                {
                    Buffer.BlockCopy(layer.weightMx, 0, weightsAndBiases, offset, layer.weightMx.Length);
                    offset += layer.weightMx.Length * 4;
                    Buffer.BlockCopy(layer.biases, 0, weightsAndBiases, offset, layer.biases.Length);
                    offset += layer.biases.Length * 4;
                }
            }
            MemoryAllocation mem_weightsAndBiases = computeFramework.GetMemoryFor(MemFlags.ReadOnly | MemFlags.CopyHostPtr, weightsAndBiases);

            MemoryAllocation mem_delta_k_vector = computeFramework.GetMemoryFor(widestLayerNeuronCount * network.layers.Count * 4, MemFlags.ReadWrite, IntPtr.Zero );

            int[] layerIdUpdateSubbuffer = new int[] { 0 };

            computeFramework.SetKernelArg(forwardPass, 0, mem_NetworkConfigParams);
            computeFramework.SetKernelArg(forwardPass, 1, mem_activationsAndZValues);
            computeFramework.SetKernelArg(forwardPass, 2, mem_InputActivations);
            computeFramework.SetKernelArg(forwardPass, 3, mem_weightsAndBiases);

            var localWorkGroupSize = new IntPtr[] { new IntPtr(8), new IntPtr(8) };
            var globalWorkSize = new IntPtr[] { new IntPtr(ExtendGlobalWorkSize(inputActivationCount, localWorkGroupSize[0].ToInt32())), new IntPtr(ExtendGlobalWorkSize(trainingSamples, localWorkGroupSize[0].ToInt32())) };

            for (int i = 0; i < network.layers.Count; i++)
            {
                if (i != 0)
                {
                    layerIdUpdateSubbuffer[0] = i;
                    computeFramework.UploadToMemory(mem_NetworkConfigParams, 0, layerIdUpdateSubbuffer, true); //Update layer index to be processed by the kernel
                }
                computeFramework.EnqueueKernel(forwardPass, globalWorkSize, localWorkGroupSize);
                // todo: run forward pass
            }

            var mem_param_gradient = computeFramework.GetMemoryFor(totalWeightAndBiasCount * 4, MemFlags.WriteOnly, IntPtr.Zero);

            // todo: run output layer pass
            computeFramework.EnqueueKernel(backwardPassOutputlayer, globalWorkSize, localWorkGroupSize);

            for (int i = network.layers.Count - 2; i >= 0; --i)
            {
                computeFramework.EnqueueKernel(backwardPassHiddenlayer, globalWorkSize, localWorkGroupSize);
            }

            float[] outputGradient = new float[totalWeightAndBiasCount];
            unsafe
            {
                fixed (float* outputPtr = outputGradient)
                {
                    computeFramework.ReadBuffer(mem_param_gradient, true, new IntPtr(0), new IntPtr(totalWeightAndBiasCount * 4), new IntPtr(outputPtr));
                }
            }

            return ret;
        }

    }
}
