using System;
using System.Collections.Generic;
using System.Threading;
using Mademy;
using Mademy.OpenCL;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace UnitTestProject1
{
    [TestClass]
    public class UnitTest1
    {
        public void CheckNetworkError(float[] a, float[] b)
        {
            float error = 0;

            if (a.Length != b.Length)
                Assert.Fail();

            for (int i = 0; i < a.Length; i++)
            {
                error += Math.Abs(a[i] - b[i]);
            }

            if (error > 0.00001f)
                Assert.Fail();
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
            Network networkCpuTrained = Network.LoadTrainingDataFromJSON(jsonData);
            Network networkOpenCLTrained = Network.LoadTrainingDataFromJSON(jsonData);

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
            Network networkCpuTrained = Network.LoadTrainingDataFromJSON(jsonData);
            Network networkOpenCLTrained = Network.LoadTrainingDataFromJSON(jsonData);

            MathLib cpuCalculator = new MathLib();
            MathLib openCLCalculator = new MathLib(ComputeDevice.GetDevices()[0]);

            float[] testInput = new float[layerConfig[0]];
            var cpuTrainedOutput = networkCpuTrained.Compute(cpuCalculator, testInput);
            var openCLTrainedOutput = networkOpenCLTrained.Compute(cpuCalculator, testInput);

            CheckNetworkError(cpuTrainedOutput, openCLTrainedOutput);
        }
    }
}
