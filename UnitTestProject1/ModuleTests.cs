using System;
using System.Collections.Generic;
using System.Threading;
using Mademy;
using Mademy.OpenCL;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace UnitTestProject1
{
    [TestClass]
    public class ModuleTests
    {
        public void CheckNetworkError(float[] a, float[] b)
        {
            double error = 0;

            if (a.Length != b.Length)
                Assert.Fail("Network output sizes do not match!");

            for (int i = 0; i < a.Length; i++)
            {
                error += Math.Abs((double)a[i] - (double)b[i]);
            }

            var meanError = (error / a.Length);
            if (meanError > 0.001) 
                Assert.Fail("Networks do not match. Error was: " + meanError);
        }

        [TestMethod]
        public void TestOpenCLTraining()
        {
            List<int> layerConfig = new List<int>();
            layerConfig.Add(10);
            layerConfig.Add(512);
            layerConfig.Add(12);
            layerConfig.Add(3);
            layerConfig.Add(51);
            layerConfig.Add(30);

            Network networkReference = Network.CreateNetworkInitRandom(layerConfig, new SigmoidActivation());
            var jsonData = networkReference.GetTrainingDataJSON();
            Network networkCpuTrained = Network.CreateNetworkFromJSON(jsonData);
            Network networkOpenCLTrained = Network.CreateNetworkFromJSON(jsonData);

            MathLib cpuCalculator = new MathLib();
            MathLib openCLCalculator = new MathLib(ComputeDevice.GetDevices()[0]);

            var rnd = new Random();
            List<TrainingSuite.TrainingData> trainingData = new List<TrainingSuite.TrainingData>();
            for (int i = 0; i < 1000; i++)
            {
                float[] input = new float[layerConfig[0]];
                float[] output = new float[layerConfig[layerConfig.Count - 1]];

                var idx = rnd.Next(0, input.Length);
                input[rnd.Next(0, input.Length)] = 1.0f;

                for (int j = 0; j < 10; j++)
                {
                    output[j*3+0] = idx * 0.1f;
                    output[j*3+1] = 1.0f - (idx * 0.1f);
                    output[j*3+2] = idx * 0.05f;
                }

                trainingData.Add(new TrainingSuite.TrainingData(input, output));
            }

            TrainingSuite suite = new TrainingSuite(trainingData);
            suite.config.epochs = 1;
            suite.config.shuffleTrainingData = false;
            suite.config.miniBatchSize = 13;

            var promise1 = networkCpuTrained.Train(cpuCalculator, suite);
            var promise2 = networkOpenCLTrained.Train(openCLCalculator, suite);

            while (!promise1.IsReady() || !promise2.IsReady())
            {
                Thread.Sleep(20);
            }

            float[] testInput = new float[layerConfig[0]];

            var cpuTrainedOutput = networkCpuTrained.Compute(cpuCalculator, testInput);
            var openCLTrainedOutput = networkOpenCLTrained.Compute(cpuCalculator, testInput);

            CheckNetworkError(cpuTrainedOutput, openCLTrainedOutput);
        }


        [TestMethod]
        public void TestOpenCLLayerCalc()
        {
            List<int> layerConfig = new List<int>();
            layerConfig.Add(10);
            layerConfig.Add(512);
            layerConfig.Add(12);
            layerConfig.Add(3);
            layerConfig.Add(51);
            layerConfig.Add(30);

            Network networkReference = Network.CreateNetworkInitRandom(layerConfig, new SigmoidActivation());
            var jsonData = networkReference.GetTrainingDataJSON();
            Network networkCpuTrained = Network.CreateNetworkFromJSON(jsonData);
            Network networkOpenCLTrained = Network.CreateNetworkFromJSON(jsonData);

            MathLib cpuCalculator = new MathLib();
            MathLib openCLCalculator = new MathLib(ComputeDevice.GetDevices()[0]);

            float[] testInput = new float[layerConfig[0]];
            var cpuTrainedOutput = networkCpuTrained.Compute(cpuCalculator, testInput);
            var openCLTrainedOutput = networkOpenCLTrained.Compute(cpuCalculator, testInput);

            CheckNetworkError(cpuTrainedOutput, openCLTrainedOutput);
        }


        [TestMethod]
        public void TestTrainingRegression()
        {
            List<int> layerConfig = new List<int>();
            layerConfig.Add(5);
            layerConfig.Add(33);
            layerConfig.Add(12);
            layerConfig.Add(51);
            layerConfig.Add(5);

            Network network = Network.CreateNetworkFromJSON(Properties.Resources.ReferenceNetwork1JSON);

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

            var promise = network.Train(new MathLib(), suite);

            while (!promise.IsReady())
            {
                Thread.Sleep(20);
            }
            #endregion

            float[] testInput = new float[] {0.3f, 0.4f, 0.6f, 0.1f, 0.5f };
            var result = network.Compute(new MathLib(), testInput);

            float[] referenceOutput = new float[] { 3.46114769E-11f, 0.139522761f, 3.66372E-05f, 0.391267f, 1.0775824E-06f };

            CheckNetworkError(result, referenceOutput); 
        }
    }
}
