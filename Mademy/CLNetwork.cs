using Newtonsoft.Json;
using OpenCL.Net;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;

namespace Mademy
{
    [Serializable]
    public class CLNetwork : ISerializable
    {
        public struct TrainingConfig
        {
            public static readonly int DontSubdivideBatches = -1;

            public int miniBatchSize;
            public float learningRate;

            public TrainingConfig CreateTrainingConfig()
            {
                var ret = new TrainingConfig();
                ret.miniBatchSize = DontSubdivideBatches;
                learningRate = 0.001f;
                return ret;
            }

            public bool UseMinibatches() { return miniBatchSize != DontSubdivideBatches; }
        };

        //Serialized data
        string name;
        string description;
        float[] network;
        int[] layerConfiguration; //Number of neurons in each layer

        //Non serialized data
        Context clContext;
        ComputeDevice clDevice = null;
        CommandQueue commandQueue;
        IMem mem_param_network, mem_param_layerConfig;
        Program clProgram;
        Kernel computeKernel, trainKernel;
        bool hasClInitialized = false;

        CLNetwork(float[] network, int[] layerConfiguration)
        {
            this.network = network;
            this.layerConfiguration = layerConfiguration;
        }

        CLNetwork(SerializationInfo info, StreamingContext context)
        {
            name = (string)info.GetValue("name", typeof(string));
            description = (string)info.GetValue("description", typeof(string));
            network = (float[])info.GetValue("network", typeof(List<float[,]>));
            layerConfiguration = (int[])info.GetValue("layerConfiguration", typeof(List<float[]>));
        }

        public void GetObjectData(SerializationInfo info, StreamingContext context)
        {
            info.AddValue("name", name, typeof(string));
            info.AddValue("description", description, typeof(string));
            info.AddValue("network", network);
            info.AddValue("layerConfiguration", layerConfiguration);
        }

        void AttachName(string _name) { name = _name; }
        void AttachDescription(string _desc) { description = _desc; }

        public void SetComputeDevice(ComputeDevice _computeDevice)
        {
            CleanupCLResources();
            clDevice = _computeDevice;
        }

        public void CleanupCLResources()
        {
            if (hasClInitialized)
            {
                Cl.ReleaseMemObject(mem_param_network);
                Cl.ReleaseMemObject(mem_param_layerConfig);
                Cl.ReleaseKernel(computeKernel);
                Cl.ReleaseKernel(trainKernel);
                Cl.ReleaseProgram(clProgram);
                Cl.ReleaseCommandQueue(commandQueue);
                Cl.ReleaseContext(clContext);
                hasClInitialized = false;
            }
        }

        public float[] Compute( float[] input)
        {
            ErrorCode err;

            if ( input.Length != layerConfiguration[0])
                throw new Exception("Invalid input size! Expected " + layerConfiguration[0]);

            if (clDevice == null)
            {
                throw new Exception("No compute device set!");
            }

            InitCL();

            //Parameter 1, input
            var mem_param1 = Cl.CreateBuffer<float>(clContext, MemFlags.ReadOnly | MemFlags.CopyHostPtr, input, out err);

            //Parameter 4, output
            int outputSize = layerConfiguration.Last() * 4;
            float[] output = new float[outputSize];
            var mem_param4 = Cl.CreateBuffer<float>(clContext, MemFlags.WriteOnly, output, out err);

            Cl.SetKernelArg(computeKernel, 0, mem_param1);
            Cl.SetKernelArg(computeKernel, 1, mem_param_network);
            Cl.SetKernelArg(computeKernel, 2, mem_param_layerConfig);
            Cl.SetKernelArg(computeKernel, 3, mem_param4);

            Event e;
            Cl.EnqueueNDRangeKernel(commandQueue, computeKernel, 1, null, new IntPtr[] { new IntPtr(10) }, null, 0, null, out e);
            Cl.EnqueueReadBuffer<float>(commandQueue, mem_param4, Bool.True, output, 0, null, out e);

            Cl.ReleaseMemObject(mem_param1);
            Cl.ReleaseMemObject(mem_param4);

            return output;
        }

        private void InitCL()
        {
            if (!hasClInitialized)
            {
                ErrorCode err;
                ContextProperty[] contextProps = new ContextProperty[] { new ContextProperty(ContextProperties.Platform, (IntPtr)clDevice.GetPlatformID()) };
                clContext = Cl.CreateContext(contextProps, 1, new Device[] { clDevice.GetDevice() }, null, IntPtr.Zero, out err);
                if (err != ErrorCode.Success) throw new Exception("Failed to create context!");

                commandQueue = Cl.CreateCommandQueue(clContext, clDevice.GetDevice(), CommandQueueProperties.None, out err);
                if (err != ErrorCode.Success) throw new Exception("Failed to create command queue!");

                //Parameter 2, network
                mem_param_network = Cl.CreateBuffer<float>(clContext, MemFlags.ReadOnly | MemFlags.CopyHostPtr, network, out err);

                //Parameter 3, layerConfig
                mem_param_layerConfig = Cl.CreateBuffer<int>(clContext, MemFlags.ReadOnly | MemFlags.CopyHostPtr, layerConfiguration, out err);

                clProgram = Cl.CreateProgramWithSource(clContext, 1, new string[] { "clsrc" }, null, out err);
                if (err != ErrorCode.Success) throw new Exception("Failed to create program!");

                Cl.BuildProgram(clProgram, 1, new Device[] { clDevice.GetDevice() }, "", null, IntPtr.Zero);
                if (err != ErrorCode.Success)
                {
                    //TODO print build log!
                    throw new Exception("Failed to build program!");
                }

                computeKernel = Cl.CreateKernel(clProgram, "computeNetwork", out err);
                if (err != ErrorCode.Success) throw new Exception("Failed to create compute kernel!");
                trainKernel = Cl.CreateKernel(clProgram, "trainNetwork", out err);
                if (err != ErrorCode.Success) throw new Exception("Failed to create train kernel!");

                hasClInitialized = true;
            }
        }

        public static CLNetwork CreateNetwork(List<float[,]> weights, List<float[]> biases)
        {
            List<float> network = new List<float>();
            int[] layerConf = new int[weights.Count + 1];

            for (int i = 0; i < weights.Count; i++)
            {
                layerConf[i] = weights[i].GetLength(0);
            }
            layerConf[weights.Count] = weights.Last().GetLength(1);


            for (int i = 0; i < weights.Count; i++)
            {
                for (int j = 0; j < weights[i].GetLength(0); j++)
                {
                    for (int k = 0; k < weights[i].GetLength(1); k++)
                    {
                        network.Add(weights[i][j, k]);
                    }
                }
                for (int j = 0; j < biases[i].Length; j++)
                {
                    network.Add(biases[i][j]);
                }
            }

            return new CLNetwork(network.ToArray(), layerConf);
        }

        public string GetTrainingDataJSON()
        {
            string output = JsonConvert.SerializeObject(this);
            return output;
        }

        public static CLNetwork LoadTrainingDataFromJSON(string jsonData)
        {
            return JsonConvert.DeserializeObject<CLNetwork>(jsonData);
        }

        public static CLNetwork CreateNetworkInitRandom(List<int> layerConfig)
        {
            List<float[,]> weights = new List<float[,]>();
            List<float[]> biases = new List<float[]>();

            for (int layId = 1; layId < layerConfig.Count; ++layId)
            {
                int prevLayerSize = layerConfig[layId - 1];
                int layerSize = layerConfig[layId];
                float[,] weightMx = new float[layerSize,prevLayerSize];
                float[] biasList = new float[layerSize];
                for (int i = 0; i < layerSize; i++)
                {
                    for (int j = 0; j < prevLayerSize; j++)
                    {
                        weightMx[i, j] = Utils.GetRandomValue();
                    }
                    biasList[i] = Utils.GetRandomValue();
                }
                weights.Add(weightMx);
                biases.Add(biasList);
            }

            return CreateNetwork(weights, biases);
        }
    }
}
