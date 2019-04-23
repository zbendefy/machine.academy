using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CLMath
{
    public class TrainingSuite
    {
        public struct TrainingConfig
        {
            public static readonly int DontSubdivideBatches = -1;
            public static readonly int AutoDetectThreads = -1;
            private static int CpuCount = 1;

            public int miniBatchSize;
            public float learningRate;
            public int numThreads;

            public static TrainingConfig CreateTrainingConfig()
            {
                var ret = new TrainingConfig();

                CpuCount = Environment.ProcessorCount;

                ret.miniBatchSize = 1000;
                ret.learningRate = 0.01f;
                ret.numThreads = AutoDetectThreads;
                return ret;
            }

            public bool UseMinibatches() { return miniBatchSize != DontSubdivideBatches; }

            internal int GetThreadCount()
            {
                return numThreads == AutoDetectThreads ? CpuCount : numThreads;
            }
        };

        public struct TrainingData
        {
            public float[] input;
            public float[] desiredOutput;

            public TrainingData(float[] input, float[] desiredOutput)
            {
                this.input = input;
                this.desiredOutput = desiredOutput;
            }
        };

        public TrainingConfig config = TrainingConfig.CreateTrainingConfig();
        public List<TrainingData> trainingData;

        public TrainingSuite(List<TrainingData> trainingDatas)
        {
            this.trainingData = trainingDatas;
        }
    }
}
