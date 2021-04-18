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
        public void TestRandomNetworkWithNoLayers()
        {
            int[] referenceLayerConf = new int[] { 3, 4 };

            var network = Network.CreateNetworkInitRandom(referenceLayerConf, new SigmoidActivation());

            float[] result = network.Compute(new float[] { 0.2f, 0.4f, 0.5f }, ComputeDeviceFactory.CreateFallbackComputeDevice() );
            Assert.AreEqual( referenceLayerConf[referenceLayerConf.Length-1], result.Length);
        }

        [TestMethod]
        public void TestRandomNetwork()
        {
            int[] referenceLayerConf = new int[] { 3, 7, 5, 4 };

            var network = Network.CreateNetworkInitRandom(referenceLayerConf, new SigmoidActivation());

            Assert.AreEqual(3, network.GetLayerConfig()[0]);
            Assert.AreEqual(7, network.GetLayerConfig()[1]);
            Assert.AreEqual(5, network.GetLayerConfig()[2]);
            Assert.AreEqual(4, network.GetLayerConfig()[3]);

            float[] result = network.Compute(new float[] { 0.2f, 0.4f, 0.5f }, ComputeDeviceFactory.CreateFallbackComputeDevice());
            Assert.AreEqual(referenceLayerConf[referenceLayerConf.Length - 1], result.Length);
        }

        [TestMethod]
        public void TestOpenCLNetworkEvaluation()
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

            ComputeDevice cpuCalculator = ComputeDeviceFactory.CreateFallbackComputeDevice();
            ComputeDevice openCLCalculator = Utils.GetFirstOpenCLDevice();

            float[] testInput = new float[layerConfig[0]];
            var cpuTrainedOutput = networkCpuTrained.Compute(testInput, cpuCalculator);
            var openCLTrainedOutput = networkOpenCLTrained.Compute(testInput, openCLCalculator);

            Utils.ValidateFloatArray(cpuTrainedOutput, openCLTrainedOutput);
        }

        [TestMethod]
        public void TestTrainingRegression_CrossEntropy_L2()
        {
            float[] referenceOutput = new float[] { 5.959933E-11f, 0.1458118f, 4.22751E-05f, 0.3619123f, 1.531221E-06f };
            Network network = Network.CreateNetworkFromJSON(Testing.Properties.Resources.ReferenceNetwork1JSON);
            Utils.TestTraining(network, ComputeDeviceFactory.CreateFallbackComputeDevice(), referenceOutput, new CrossEntropyErrorFunction(), TrainingConfig.Regularization.L2, 10.0f, 0.01f);
        }

        [TestMethod]
        public void TestTrainingRegression_CrossEntropy_L1()
        {
            float[] referenceOutput = new float[] { 2.571606E-11f, 0.1361623f, 3.719596E-05f, 0.3950554f, 8.929147E-07f };
            Network network = Network.CreateNetworkFromJSON(Testing.Properties.Resources.ReferenceNetwork1JSON);
            Utils.TestTraining(network, ComputeDeviceFactory.CreateFallbackComputeDevice(), referenceOutput, new CrossEntropyErrorFunction(), TrainingConfig.Regularization.L1, 10.0f, 0.01f);
        }

        [TestMethod]
        public void TestTrainingRegression_CrossEntropy()
        {
            float[] referenceOutput = new float[] { 3.45829E-11f, 0.139514f, 3.661705E-05f, 0.3912812f, 1.076772E-06f };
            Network network = Network.CreateNetworkFromJSON(Testing.Properties.Resources.ReferenceNetwork1JSON);
            Utils.TestTraining(network, ComputeDeviceFactory.CreateFallbackComputeDevice(), referenceOutput, new CrossEntropyErrorFunction(), TrainingConfig.Regularization.None, 0.0f, 0.01f);
        }

        [TestMethod]
        public void TestTrainingRegression_MeanSquared_L2()
        {
            float[] referenceOutput = new float[] { 2.466907E-12f, 0.2984998f, 3.949761E-08f, 0.000338129f, 4.667237E-06f };
            Network network = Network.CreateNetworkFromJSON(Testing.Properties.Resources.ReferenceNetwork1JSON);
            Utils.TestTraining(network, ComputeDeviceFactory.CreateFallbackComputeDevice(), referenceOutput, new MeanSquaredErrorFunction(), TrainingConfig.Regularization.L2, 10.0f, 0.01f);
        }

        [TestMethod]
        public void TestTrainingRegression_MeanSquared_L1()
        {
            float[] referenceOutput = new float[] { 1.059097E-12f, 0.3169f, 3.026366E-08f, 0.0003242506f, 3.843542E-06f };
            Network network = Network.CreateNetworkFromJSON(Testing.Properties.Resources.ReferenceNetwork1JSON);
            Utils.TestTraining(network, ComputeDeviceFactory.CreateFallbackComputeDevice(), referenceOutput, new MeanSquaredErrorFunction(), TrainingConfig.Regularization.L1, 10.0f, 0.01f);
        }

        [TestMethod]
        public void TestTrainingRegression_MeanSquared()
        {
            float[] referenceOutput = new float[] { 1.485432E-12f, 0.3179092f, 3.043587E-08f, 0.0003209518f, 4.003032E-06f };
            Network network = Network.CreateNetworkFromJSON(Testing.Properties.Resources.ReferenceNetwork1JSON);
            Utils.TestTraining(network, ComputeDeviceFactory.CreateFallbackComputeDevice(), referenceOutput, new MeanSquaredErrorFunction(), TrainingConfig.Regularization.None, 0.0f, 0.01f);
        }

        [TestMethod]
        public void TestTrainingRegression_OpenCL_CrossEntropy_L2()
        {
            float[] referenceOutput = new float[] { 5.959933E-11f, 0.1458118f, 4.22751E-05f, 0.3619123f, 1.531221E-06f };
            Network network = Network.CreateNetworkFromJSON(Testing.Properties.Resources.ReferenceNetwork1JSON);
            Utils.TestTraining(network, Utils.GetFirstOpenCLDevice(), referenceOutput, new CrossEntropyErrorFunction(), TrainingConfig.Regularization.L2, 10.0f, 0.01f);
        }

        [TestMethod]
        public void TestTrainingRegression_OpenCL_CrossEntropy_L1()
        {
            float[] referenceOutput = new float[] { 2.571606E-11f, 0.1361623f, 3.719596E-05f, 0.3950554f, 8.929147E-07f };
            Network network = Network.CreateNetworkFromJSON(Testing.Properties.Resources.ReferenceNetwork1JSON);
            Utils.TestTraining(network, Utils.GetFirstOpenCLDevice(), referenceOutput, new CrossEntropyErrorFunction(), TrainingConfig.Regularization.L1, 10.0f, 0.01f);
        }

        [TestMethod]
        public void TestTrainingRegression_OpenCL_CrossEntropy()
        {
            float[] referenceOutput = new float[] { 3.45829E-11f, 0.139514f, 3.661705E-05f, 0.3912812f, 1.076772E-06f };
            Network network = Network.CreateNetworkFromJSON(Testing.Properties.Resources.ReferenceNetwork1JSON);
            Utils.TestTraining(network, Utils.GetFirstOpenCLDevice(), referenceOutput, new CrossEntropyErrorFunction(), TrainingConfig.Regularization.None, 0.0f, 0.01f);
        }

        [TestMethod]
        public void TestTrainingRegression_OpenCL_MeanSquared_L2()
        {
            float[] referenceOutput = new float[] { 2.466907E-12f, 0.2984998f, 3.949761E-08f, 0.000338129f, 4.667237E-06f };
            Network network = Network.CreateNetworkFromJSON(Testing.Properties.Resources.ReferenceNetwork1JSON);
            Utils.TestTraining(network, Utils.GetFirstOpenCLDevice(), referenceOutput, new MeanSquaredErrorFunction(), TrainingConfig.Regularization.L2, 10.0f, 0.01f);
        }

        [TestMethod]
        public void TestTrainingRegression_OpenCL_MeanSquared_L1()
        {
            float[] referenceOutput = new float[] { 1.059097E-12f, 0.3169f, 3.026366E-08f, 0.0003242506f, 3.843542E-06f };
            Network network = Network.CreateNetworkFromJSON(Testing.Properties.Resources.ReferenceNetwork1JSON);
            Utils.TestTraining(network, Utils.GetFirstOpenCLDevice(), referenceOutput, new MeanSquaredErrorFunction(), TrainingConfig.Regularization.L1, 10.0f, 0.01f);
        }

        [TestMethod]
        public void TestTrainingRegression_OpenCL_MeanSquared()
        {
            float[] referenceOutput = new float[] { 1.485432E-12f, 0.3179092f, 3.043587E-08f, 0.0003209518f, 4.003032E-06f };
            Network network = Network.CreateNetworkFromJSON(Testing.Properties.Resources.ReferenceNetwork1JSON);
            Utils.TestTraining(network, Utils.GetFirstOpenCLDevice(), referenceOutput, new MeanSquaredErrorFunction(), TrainingConfig.Regularization.None, 0.0f, 0.01f);
        }

        [TestMethod]
        public void TestTrainingOpenCL_CrossEntropy_L2()
        {
            Utils.TestOpenCLTrainingWithConfig(new CrossEntropyErrorFunction(), TrainingConfig.Regularization.L2, 0.01f, 0.01f);
            Utils.TestOpenCLTrainingWithConfig(new CrossEntropyErrorFunction(), TrainingConfig.Regularization.L2, 0.01f, 0.0f);
            Utils.TestOpenCLTrainingWithConfig(new CrossEntropyErrorFunction(), TrainingConfig.Regularization.L2, 0.0f, 0.01f);
            Utils.TestOpenCLTrainingWithConfig(new CrossEntropyErrorFunction(), TrainingConfig.Regularization.L2, 0.0f, 0.0f);
        }

        [TestMethod]
        public void TestTrainingOpenCL_CrossEntropy_L1()
        {
            Utils.TestOpenCLTrainingWithConfig(new CrossEntropyErrorFunction(), TrainingConfig.Regularization.L1, 0.01f, 0.01f);
            Utils.TestOpenCLTrainingWithConfig(new CrossEntropyErrorFunction(), TrainingConfig.Regularization.L1, 0.01f, 0.0f);
            Utils.TestOpenCLTrainingWithConfig(new CrossEntropyErrorFunction(), TrainingConfig.Regularization.L1, 0.0f, 0.01f);
            Utils.TestOpenCLTrainingWithConfig(new CrossEntropyErrorFunction(), TrainingConfig.Regularization.L1, 0.1f, 0.0f);
        }

        [TestMethod]
        public void TestTrainingOpenCL_CrossEntropy_NpRegularization()
        {
            Utils.TestOpenCLTrainingWithConfig(new CrossEntropyErrorFunction(), TrainingConfig.Regularization.None, 0.0f, 0.01f);
            Utils.TestOpenCLTrainingWithConfig(new CrossEntropyErrorFunction(), TrainingConfig.Regularization.None, 0.0f, 0.00f);
        }

        [TestMethod]
        public void TestTrainingOpenCL_MSE_L2()
        {
            Utils.TestOpenCLTrainingWithConfig(new MeanSquaredErrorFunction(), TrainingConfig.Regularization.L2, 0.01f, 0.01f);
            Utils.TestOpenCLTrainingWithConfig(new MeanSquaredErrorFunction(), TrainingConfig.Regularization.L2, 0.01f, 0.0f);
            Utils.TestOpenCLTrainingWithConfig(new MeanSquaredErrorFunction(), TrainingConfig.Regularization.L2, 0.0f, 0.01f);
            Utils.TestOpenCLTrainingWithConfig(new MeanSquaredErrorFunction(), TrainingConfig.Regularization.L2, 0.0f, 0.0f);
        }

        [TestMethod]
        public void TestTrainingOpenCL_MSE_L1()
        {
            Utils.TestOpenCLTrainingWithConfig(new MeanSquaredErrorFunction(), TrainingConfig.Regularization.L1, 0.01f, 0.01f);
            Utils.TestOpenCLTrainingWithConfig(new MeanSquaredErrorFunction(), TrainingConfig.Regularization.L1, 0.01f, 0.0f);
            Utils.TestOpenCLTrainingWithConfig(new MeanSquaredErrorFunction(), TrainingConfig.Regularization.L1, 0.0f, 0.01f);
            Utils.TestOpenCLTrainingWithConfig(new MeanSquaredErrorFunction(), TrainingConfig.Regularization.L1, 0.1f, 0.0f);
        }

        [TestMethod]
        public void TestTrainingOpenCL_MSE_NpRegularization()
        {
            Utils.TestOpenCLTrainingWithConfig(new MeanSquaredErrorFunction(), TrainingConfig.Regularization.None, 0.0f, 0.01f);
            Utils.TestOpenCLTrainingWithConfig(new MeanSquaredErrorFunction(), TrainingConfig.Regularization.None, 0.0f, 0.0f);
        }

        [TestMethod]
        public void TestTrainingOpenCL_MixedActivationFunction()
        {
            Utils.TestOpenCLTrainingWithConfig(new MeanSquaredErrorFunction(), TrainingConfig.Regularization.None, 0.0f, 0.01f, true);
            Utils.TestOpenCLTrainingWithConfig(new CrossEntropyErrorFunction(), TrainingConfig.Regularization.None, 0.0f, 0.1f, true);
            Utils.TestOpenCLTrainingWithConfig(new CrossEntropyErrorFunction(), TrainingConfig.Regularization.L1, 0.1f, 0.0f, true);
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

            float[] outputRef = networkReference.Compute(testInput, ComputeDeviceFactory.CreateFallbackComputeDevice());
            float[] outputJS = networkFromJSON.Compute(testInput, ComputeDeviceFactory.CreateFallbackComputeDevice());

            Utils.ValidateFloatArray(outputRef, outputJS, 0.00000001);
        }
    }
}
