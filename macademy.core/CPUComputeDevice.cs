using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Macademy
{

    internal class CPUComputeDeviceDesc : ComputeDeviceDesc
    {
        public override ComputeDevice CreateDevice()
        {
            return new CPUComputeDevice(this);
        }

        public override string GetDeviceAccessType()
        {
            return "CPU";
        }

        public override int GetDeviceCoreCount()
        {
            return 1;
        }

        public override long GetDeviceMemorySize()
        {
            return 0L;
        }

        public override string GetDeviceName()
        {
            return "Generic CPU";
        }
    }

    internal class CPUComputeDevice : ComputeDevice
    {
        internal CPUComputeDevice(ComputeDeviceDesc desc) : base(desc)
        {
        }

        public override List<List<NeuronData>> CalculateAccumulatedGradientForMinibatch(Network network, TrainingSuite suite, int trainingDataBegin, int trainingDataEnd)
        {
            int trainingSamples = trainingDataEnd - trainingDataBegin;
            var ret = Utils.CreateGradientVector(network);
            for (int i = trainingDataBegin; i < trainingDataEnd; i++)
            {
                CalculateGradientForSingleTrainingExample(network, suite.config.costFunction, ref ret, suite.trainingData[i].input, suite.trainingData[i].desiredOutput);
            }
            return ret;
        }

        public override float[] CalculateLayer(float[,] weightMx, float[] bias, float[] prevActivations, IActivationFunction activationFunction)
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

                ret[m] = activationFunction.Calculate(acc);
            }
            return ret;
        }

        public override void FlushWorkingCache()
        {
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
            var activationFunction = network.layers.Last().activationFunction;
            for (int i = 0; i < lastLayerNeuronCount; i++)
            {
                float outputValue = activations.Last()[i];
                float delta_k = errorFunction.CalculateDelta(zValues.Last()[i], outputValue, desiredOutput[i], activationFunction);

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
                deltak *= network.layers[L].activationFunction.CalculatePrime(zValues[L][i]);
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

        public override void Uninitialize()
        {
        }

        public static List<ComputeDeviceDesc> GetDevices()
        {
            List<ComputeDeviceDesc> ret = new List<ComputeDeviceDesc>();
            CPUComputeDeviceDesc cpuDevice = new CPUComputeDeviceDesc();
            ret.Add(cpuDevice);
            return ret;
        }
    }
}
