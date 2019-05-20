using System;
using System.Collections.Generic;
using System.Threading;
using Macademy;
using Macademy.OpenCL;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace ModuleTests
{
    [TestClass]
    public class ModuleTests
    {
        [TestMethod]
        public void TestRandomNetwork()
        {
            int[] referenceLayerConf = new int[] { 3, 7, 5, 4 };

            var network = Network.CreateNetworkInitRandom(referenceLayerConf, new SigmoidActivation());

            Assert.AreEqual(3, network.GetLayerConfig()[0]);
            Assert.AreEqual(7, network.GetLayerConfig()[1]);
            Assert.AreEqual(5, network.GetLayerConfig()[2]);
            Assert.AreEqual(4, network.GetLayerConfig()[3]);

            float[] result = network.Compute(new float[] { 0.2f, 0.4f, 0.5f });
            Assert.AreEqual( referenceLayerConf[referenceLayerConf.Length-1], result.Length);
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
            float[] referenceOutput = new float[] { 3.46114769E-11f, 0.139522761f, 3.66372E-05f, 0.391267f, 1.0775824E-06f };
            Utils.TestTraining(referenceOutput, new CrossEntropyErrorFunction(), TrainingSuite.TrainingConfig.Regularization.L2, 0.01f, 0.01f);
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
