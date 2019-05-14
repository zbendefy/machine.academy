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
        private DeviceConfig deviceConfig;
        private static readonly string calcLayerKernel = "calcSingleLayer";
        private static readonly string forwardPass = "trainingForwardPass";
        private static readonly string backwardPassKernel = "trainingBackwardPass";

        public class DeviceConfig
        {
            public int idealWorkgroupSizeX = 8;
            public int idealWorkgroupSizeY = 8;
            public string compileOptions = "-cl-mad-enable -cl-no-signed-zeros";
        }

        public MathLib(ComputeDevice clDevice = null, DeviceConfig _deviceConfig = null)
        {
            deviceConfig = _deviceConfig;
            if (deviceConfig == null)
                deviceConfig = new DeviceConfig();

            if (clDevice != null)
                computeFramework = new ComputeFramework(clDevice, new string[] { CLSourceProvider.ReadSourceFile() }, new string[] { calcLayerKernel, forwardPass, backwardPassKernel }, deviceConfig.compileOptions);
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

        internal unsafe float[] CalculateLayer(float[,] weightMx, float[] bias, float[] prevActivations, IActivationFunction sigmoidFunction)
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
            float[] output = new float[matrixRows];
            int[] configParams = new int[] { /*rows: */weightMx.GetLength(0), /*cols: */weightMx.GetLength(1), /*ApplySigmoid*/ sigmoidFunction.GetOpenCLFunctionId() };

            fixed (int* configPtr = configParams)
            {
                fixed (float* weightArrayPtr = weightMx, biasPtr = bias, prevActivationPtr = prevActivations)
                {
                    MemoryAllocation mem_param_weightMx, mem_param_bias, mem_param_prevActivation, mem_param_config, mem_param_output;
                    mem_param_weightMx = computeFramework.GetMemoryFor(weightMx.Length * 4, MemFlags.ReadOnly | MemFlags.CopyHostPtr, new IntPtr(weightArrayPtr));
                    mem_param_bias = computeFramework.GetMemoryFor(bias.Length * 4, MemFlags.ReadOnly | MemFlags.CopyHostPtr, new IntPtr(biasPtr));
                    mem_param_prevActivation = computeFramework.GetMemoryFor(prevActivations.Length * 4, MemFlags.ReadOnly | MemFlags.CopyHostPtr, new IntPtr(prevActivationPtr));
                    mem_param_config = computeFramework.GetMemoryFor(configParams.Length * 4, MemFlags.ReadOnly | MemFlags.CopyHostPtr, new IntPtr(configPtr));
                    mem_param_output = computeFramework.GetMemoryFor(matrixRows * 4, MemFlags.WriteOnly, IntPtr.Zero);

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
                        computeFramework.ReadBuffer(mem_param_output, true, IntPtr.Zero, new IntPtr(matrixRows * 4), new IntPtr(outputPtr));
                    }
                }
            }

            computeFramework.UnuseMemoryAllocations();

            return output;
        }

        public void CleanupResources()
        {
            if (computeFramework != null)
                computeFramework.CleanupCLResources();
        }

        private void CalculateGradientForSingleTrainingExample(Network network, IErrorFunction errorFunction, ref List<List<NeuronData>> intermediateResults, float[] trainingInput, float[] trainingDesiredOutput)
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
                    gradientDataItem.weights[j] += delta_k * prevActivations[j];
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

            for (int i = 0; i < layerNeuronCount; ++i)
            {
                float deltak = 0;
                //Assert(delta_k_vector.Count == layers[L + 1].weightMx.GetLength(0));
                for (int k = 0; k < delta_k_vector.Count; ++k)
                {
                    deltak += delta_k_vector[k] * network.layers[L + 1].weightMx[k, i];
                }
                deltak *= network.activationFunction.CalculatePrime(zValues[L][i]);
                newGammak.Add(deltak);

                //Assert(gradientData[i].weights.Length == prevLayerActivations.Length);
                var gradientDataItem = gradientData[i];
                for (int j = 0; j < layerWeightCount; ++j)
                {
                    gradientDataItem.weights[j] += deltak * (prevLayerActivations[j]);
                }
                gradientDataItem.bias += deltak;
            }

            delta_k_vector = newGammak;
        }

        internal void FlushWorkingCache()
        {
            if (HasComputeFramework())
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
        internal unsafe List<List<NeuronData>> CalculateAccumulatedGradientForMinibatch(Network network, TrainingSuite suite, int trainingDataBegin, int trainingDataEnd)
        {
            int trainingSamples = trainingDataEnd - trainingDataBegin;
            var ret = Utils.CreateGradientVector(network);

            if (!HasComputeFramework()) //CPU fallback
            {
                for (int i = trainingDataBegin; i < trainingDataEnd; i++)
                {
                    CalculateGradientForSingleTrainingExample(network, suite.config.costFunction, ref ret, suite.trainingData[i].input, suite.trainingData[i].desiredOutput);
                }
                return ret;
            }


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
                    MemoryAllocation mem_NetworkConfigParams = computeFramework.GetMemoryFor(networkConfigParams.Length * 4, MemFlags.ReadOnly | MemFlags.CopyHostPtr, new IntPtr(networkConfigParamsPtr));

                    for (int i = 0; i < trainingSamples; ++i)
                        Buffer.BlockCopy(suite.trainingData[trainingDataBegin + i].input, 0, inputParameters, i * inputActivationCount * 4, inputActivationCount * 4);
                    MemoryAllocation mem_InputActivations = computeFramework.GetMemoryFor(inputParameters.Length * 4, MemFlags.ReadOnly | MemFlags.CopyHostPtr, new IntPtr(inputParametersPtr));

                    ///Contains the whole network's activation values, and Z values for each training sample
                    ///Memory layout for one layer is like this: [...input values...][...first layer's activations...][...second layer's activations]...[last layer's activations][first layer's z values][second layer's zvalues]...[last layer's z values]
                    ///After that, the next layer's same values are there
                    MemoryAllocation mem_activationsAndZValues = computeFramework.GetMemoryFor(totalActivationCount * trainingSamples * 2 * 4, MemFlags.ReadWrite, IntPtr.Zero);

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
                    MemoryAllocation mem_weightsAndBiases = computeFramework.GetMemoryFor(weightsAndBiases.Length * 4, MemFlags.ReadOnly | MemFlags.CopyHostPtr, new IntPtr(weightsAndBiasesPtr));

                    //delta_k_vector is double buffered (hence the * 2). In a pass, the previous delta_k values are read, and the next ones are written
                    //Memory layout is: [delta_k_vector buffer1 of trainingSample0][delta_k_vector buffer2 of trainingSample0] [delta_k_vector buffer1 of trainingSample1][delta_k_vector buffer2 of trainingSample1] ...
                    MemoryAllocation mem_delta_k_vector = computeFramework.GetMemoryFor(Math.Max(1, delta_k_vectorSize * trainingSamples * 2 * 4), MemFlags.ReadWrite, IntPtr.Zero);

                    computeFramework.SetKernelArg(forwardPass, 0, mem_NetworkConfigParams);
                    computeFramework.SetKernelArg(forwardPass, 1, mem_activationsAndZValues);
                    computeFramework.SetKernelArg(forwardPass, 2, mem_InputActivations);
                    computeFramework.SetKernelArg(forwardPass, 3, mem_weightsAndBiases);

                    int[] layerIndexBuffer = new int[network.layers.Count];
                    for (int i = 0; i < layerIndexBuffer.Length; ++i)
                        layerIndexBuffer[i] = i;

                    var localWorkGroupSize = new IntPtr[] { new IntPtr(deviceConfig.idealWorkgroupSizeX), new IntPtr(deviceConfig.idealWorkgroupSizeY) };
                    var globalWorkSize = new IntPtr[] { new IntPtr(0)
                , new IntPtr(ExtendGlobalWorkSize(trainingSamples, localWorkGroupSize[1].ToInt32())) };

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
                    for (int i = 0; i < trainingSamples; i++)
                        Buffer.BlockCopy(suite.trainingData[trainingDataBegin + i].desiredOutput, 0, desiredOutputs, i * desiredOutputByteSizePerTrainigSample, desiredOutputByteSizePerTrainigSample);
                    var mem_desired_outputs = computeFramework.GetMemoryFor(desiredOutputs.Length * 4, MemFlags.ReadOnly | MemFlags.CopyHostPtr, new IntPtr(desiredOutputsPtr));

                    var mem_param_gradient = computeFramework.GetMemoryFor(outputGradient.Length * 4, MemFlags.ReadWrite | MemFlags.CopyHostPtr, new IntPtr(outputGradientPtr));

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

                    computeFramework.ReadBuffer(mem_param_gradient, true, new IntPtr(0), new IntPtr(mem_param_gradient.bufferSizeInBytes), new IntPtr(outputGradientPtr));
                }
            }

            computeFramework.UnuseMemoryAllocations();

            /*for (int i = 0; i < outputGradient.Length; i++)
            {
                if (float.IsNaN(outputGradient[i]))
                {
                    Console.WriteLine("problem" );
                }
            }*/

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

            /*float error = _Debug_CalculateErrorOfOpenCLGradient(network, suite, trainingDataBegin, trainingDataEnd, ret);
            if (error > 0.001)
                return null; */

            return ret;
        }


        private float _Debug_CalculateErrorOfOpenCLGradient(Network network, TrainingSuite suite, int trainingDataBegin, int trainingDataEnd, List<List<NeuronData>> openclResults)
        {
            var tempcomp = computeFramework;
            computeFramework = null;
            var retCPU = this.CalculateAccumulatedGradientForMinibatch(network, suite, trainingDataBegin, trainingDataEnd);
            computeFramework = tempcomp;
            double error = 0;
            double divisor = 0;
            for (int i = 0; i < retCPU.Count; i++)
            {
                for (int j = 0; j < retCPU[i].Count; j++)
                {
                    for (int k = 0; k < retCPU[i][j].weights.Length; k++)
                    {
                        error += retCPU[i][j].weights[k] - openclResults[i][j].weights[k];
                    }
                    error += retCPU[i][j].bias - openclResults[i][j].bias;
                    divisor += retCPU[i][j].weights.Length + 1;
                }
            }

            return (float)(error / divisor);
        }
    }
}
