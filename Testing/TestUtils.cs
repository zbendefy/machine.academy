using Macademy;
using Macademy.OpenCL;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ModuleTests
{
    public static class Utils
    {
        public static ComputeDevice GetFirstOpenCLDevice()
        {
            foreach (var item in ComputeDeviceFactory.GetComputeDevices())
            {
                if (item.GetDeviceAccessType().ToLower() == "opencl")
                {
                    return ComputeDeviceFactory.CreateComputeDevice(item);
                }
            }
            return null;
        }

        public static void TestTraining( Network network, ComputeDevice device, float[] referenceOutput, IErrorFunction errorFunc, TrainingConfig.Regularization regularization, float regularizationLambda, float learningRate)
        {
            List<int> layerConfig = new List<int>();
            layerConfig.Add(5);
            layerConfig.Add(33);
            layerConfig.Add(12);
            layerConfig.Add(51);
            layerConfig.Add(5);

            #region Training
            List<TrainingSuite.TrainingData> trainingData = new List<TrainingSuite.TrainingData>();
            for (int i = 0; i < 1000; i++)
            {
                float[] input = new float[layerConfig[0]];
                float[] desiredOutput = new float[layerConfig[layerConfig.Count - 1]];

                input[(i * 13426) % 5] = 1.0f;
                desiredOutput[(i * 13426) % 5] = 1.0f;

                trainingData.Add(new TrainingSuite.TrainingData(input, desiredOutput));
            }

            TrainingSuite suite = new TrainingSuite(trainingData);
            suite.config.epochs = 2;
            suite.config.shuffleTrainingData = false;
            suite.config.miniBatchSize = 13;

            suite.config.costFunction = errorFunc;
            suite.config.regularization = regularization;
            suite.config.regularizationLambda = regularizationLambda;
            suite.config.learningRate = learningRate;

            var promise = network.Train(suite, device);

            promise.Await();
            #endregion

            float[] testInput = new float[] { 0.3f, 0.4f, 0.6f, 0.1f, 0.5f };
            var result = network.Compute(testInput, device);
            
            Utils.ValidateFloatArray(referenceOutput, result);
        }

        public static void TestOpenCLTrainingWithConfig(IErrorFunction errorFunc, TrainingConfig.Regularization regularization, float regularizationLambda, float learningRate, bool mix_activations = false)
        {
            IActivationFunction alternateActivation = new SigmoidActivation();
            if(mix_activations)
            {
                alternateActivation = new ReLUActivation();
            }

            int input_neurons = 10;
            var layer_config = new List<Tuple<IActivationFunction, int>>();
            layer_config.Add(new Tuple<IActivationFunction, int>(new SigmoidActivation(), 512));
            layer_config.Add(new Tuple<IActivationFunction, int>(alternateActivation, 12));
            layer_config.Add(new Tuple<IActivationFunction, int>(new SigmoidActivation(), 3));
            layer_config.Add(new Tuple<IActivationFunction, int>(alternateActivation, 51));
            layer_config.Add(new Tuple<IActivationFunction, int>(new SigmoidActivation(), 30));

            Network networkReference = Network.CreateNetworkInitRandom(input_neurons, layer_config);
            var jsonData = networkReference.ExportToJSON();
            Network networkCpuTrained = Network.CreateNetworkFromJSON(jsonData);
            Network networkOpenCLTrained = Network.CreateNetworkFromJSON(jsonData);

            var cpuCalculator = ComputeDeviceFactory.CreateFallbackComputeDevice();
            var openCLCalculator = GetFirstOpenCLDevice();

            var rnd = new Random();
            List<TrainingSuite.TrainingData> trainingData = new List<TrainingSuite.TrainingData>();
            for (int i = 0; i < 1000; i++)
            {
                float[] input = new float[input_neurons];
                float[] output = new float[layer_config.Last().Item2];

                var idx = rnd.Next(0, input.Length);
                input[rnd.Next(0, input.Length)] = 1.0f;

                for (int j = 0; j < 10; j++)
                {
                    output[j * 3 + 0] = idx * 0.1f;
                    output[j * 3 + 1] = 1.0f - (idx * 0.1f);
                    output[j * 3 + 2] = idx * 0.05f;
                }

                trainingData.Add(new TrainingSuite.TrainingData(input, output));
            }

            TrainingSuite suite = new TrainingSuite(trainingData);
            suite.config.epochs = 1;
            suite.config.shuffleTrainingData = false;
            suite.config.miniBatchSize = 13;

            suite.config.costFunction = errorFunc;
            suite.config.regularization = regularization;
            suite.config.regularizationLambda = regularizationLambda;
            suite.config.learningRate = learningRate;

            var promise1 = networkCpuTrained.Train(suite, cpuCalculator);
            var promise2 = networkOpenCLTrained.Train(suite, openCLCalculator);

            promise1.Await();
            promise2.Await();

            Assert.IsTrue(promise1.IsReady() && promise2.IsReady());

            float[] testInput = new float[input_neurons];

            var cpuTrainedOutput = networkCpuTrained.Compute(testInput, cpuCalculator);
            var openCLTrainedOutput = networkOpenCLTrained.Compute(testInput, cpuCalculator);

            ValidateFloatArray(cpuTrainedOutput, openCLTrainedOutput);
        }

        public static void ValidateFloatArray(float[] expected, float[] actual, double errorThreshold = 0.00001)
        {
            double error = 0;

            if (expected.Length != actual.Length)
                Assert.Fail(String.Format("Array sizes do not match! Expected size: {0}. Got: {1}", expected.Length, actual.Length));

            for (int i = 0; i < expected.Length; i++)
            {
                error += Math.Abs((double)expected[i] - (double)actual[i]);
            }

            var meanError = (error / expected.Length);
            if (meanError > errorThreshold)
            {
                Assert.Fail(String.Format("Arrays do not match. Error was: {0}. Expected: [{1}]  Got: [{2}]", meanError, string.Join(", ", expected), string.Join(", ", actual)));
            }
        }

        public static void ValidateGradient(List<List<NeuronData>> expected, List<List<NeuronData>> actual, double error)
        {
            if (expected.Count != actual.Count)
                Assert.Fail(String.Format("Layer count do not match! Expected size: {0}. Got: {1}", expected.Count, actual.Count));

            for (int i = 0; i < expected.Count; i++)
            {
                if (expected[i].Count != actual[i].Count)
                    Assert.Fail(String.Format("Layer #{0} sizes do not match! Expected size: {1}. Got: {2}", i, expected[i].Count, actual[i].Count));

                for (int j = 0; j < expected[i].Count; j++)
                {
                    var expected_neuron_data = expected[i][j];
                    var actual_neuron_data = actual[i][j];

                    if (expected_neuron_data.bias != actual_neuron_data.bias)
                    {
                        Assert.Fail(String.Format("Layer #{0}, neuron #{1} bias does not match! Expected: {2}. Got: {3}", i, j, expected[i][j].bias, actual[i][j].bias));
                    }

                    ValidateFloatArray(expected_neuron_data.weights, actual_neuron_data.weights);
                }
            }
        }
    }
}
