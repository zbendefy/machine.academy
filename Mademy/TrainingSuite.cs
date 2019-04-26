using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Mademy
{
    public class TrainingSuite
    {
        public struct TrainingConfig
        {
            public static readonly int DontSubdivideBatches = -1;

            public enum Regularization { None, L1, L2 }

            public int miniBatchSize;
            public float learningRate;
            public int epochs;
            public bool shuffleTrainingData;
            public IErrorFunction costFunction;
            public Regularization regularization;
            public float regularizationLambda;

            public static TrainingConfig CreateTrainingConfig()
            {
                var ret = new TrainingConfig();

                ret.miniBatchSize = 1000;
                ret.learningRate = 0.01f;
                ret.epochs = 1;
                ret.shuffleTrainingData = true;
                ret.costFunction = new CrossEntropy();
                ret.regularization = Regularization.L2;
                ret.regularizationLambda = 0.01f;
                return ret;
            }

            public bool UseMinibatches() { return miniBatchSize != DontSubdivideBatches; }
        };

        public class TrainingData
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
