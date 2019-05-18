using System;
using System.Collections.Generic;
using System.Threading;
using Macademy;
using Macademy.OpenCL;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace UnitTestProject1
{
    


    [TestClass]
    public class ModuleTests
    {
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

            Network networkReference = Network.CreateNetworkInitRandom(layerConfig.ToArray(), new SigmoidActivation());
            var jsonData = networkReference.ExportToJSON();
            Network networkCpuTrained = Network.CreateNetworkFromJSON(jsonData);
            Network networkOpenCLTrained = Network.CreateNetworkFromJSON(jsonData);

            Calculator cpuCalculator = new Calculator();
            Calculator openCLCalculator = new Calculator(ComputeDevice.GetDevices()[0]);

            float[] testInput = new float[layerConfig[0]];
            var cpuTrainedOutput = networkCpuTrained.Compute(testInput, cpuCalculator);
            var openCLTrainedOutput = networkOpenCLTrained.Compute(testInput, cpuCalculator);

            Utils.CheckNetworkError(cpuTrainedOutput, openCLTrainedOutput);
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

            var promise = network.Train(suite, new Calculator());

            while (!promise.IsReady())
            {
                Thread.Sleep(20);
            }
            #endregion

            float[] testInput = new float[] {0.3f, 0.4f, 0.6f, 0.1f, 0.5f };
            var result = network.Compute( testInput, new Calculator());

            float[] referenceOutput = new float[] { 3.46114769E-11f, 0.139522761f, 3.66372E-05f, 0.391267f, 1.0775824E-06f };

            Utils.CheckNetworkError(result, referenceOutput);
        }

        [TestMethod]
        public void TestTrainingOpenCL_CrossEntropy_L2()
        {
            Utils.TestOpenCLTrainingWithConfig(new CrossEntropyErrorFunction(), TrainingSuite.TrainingConfig.Regularization.L2, 0.01f, 0.01f);
        }

        [TestMethod]
        public void TestTrainingOpenCL_CrossEntropy_L1()
        {
            Utils.TestOpenCLTrainingWithConfig(new CrossEntropyErrorFunction(), TrainingSuite.TrainingConfig.Regularization.L1, 0.01f, 0.01f);
        }

        [TestMethod]
        public void TestTrainingOpenCL_CrossEntropy_NpRegularization()
        {
            Utils.TestOpenCLTrainingWithConfig(new CrossEntropyErrorFunction(), TrainingSuite.TrainingConfig.Regularization.None, 0.01f, 0.01f);
        }

        [TestMethod]
        public void TestTrainingOpenCL_MSE_L2()
        {
            Utils.TestOpenCLTrainingWithConfig(new MeanSquaredErrorFunction(), TrainingSuite.TrainingConfig.Regularization.L2, 0.01f, 0.01f);
        }

        [TestMethod]
        public void TestTrainingOpenCL_MSE_L1()
        {
            Utils.TestOpenCLTrainingWithConfig(new MeanSquaredErrorFunction(), TrainingSuite.TrainingConfig.Regularization.L1, 0.01f, 0.01f);
        }

        [TestMethod]
        public void TestTrainingOpenCL_MSE_NpRegularization()
        {
            Utils.TestOpenCLTrainingWithConfig(new MeanSquaredErrorFunction(), TrainingSuite.TrainingConfig.Regularization.None, 0.01f, 0.01f);
        }

        [TestMethod]
        public void TestJSONExportImport()
        {
            List<int> layerConfig = new List<int>();
            layerConfig.Add(5);
            layerConfig.Add(33);
            layerConfig.Add(12);
            layerConfig.Add(51);
            layerConfig.Add(5);

            float[] testInput = new float[] { 0.1f, 0.5f, 0.2f, 0.14f, 0.54f };

            Network networkReference = Network.CreateNetworkInitRandom(layerConfig.ToArray(), new SigmoidActivation());
            Network networkFromJSON = Network.CreateNetworkFromJSON(networkReference.ExportToJSON());

            float[] outputRef = networkReference.Compute(testInput);
            float[] outputJS = networkFromJSON.Compute(testInput);

            Utils.CheckNetworkError(outputRef, outputJS);
        }
    }
}
