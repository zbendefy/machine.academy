using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Macademy
{

    internal class CUDAComputeDeviceDesc : ComputeDeviceDesc
    {
        public override ComputeDevice CreateDevice()
        {
            return new CUDADevice(this);
        }

        public override string GetDeviceAccessType()
        {
            return "CUDA";
        }

        public override int GetDeviceCoreCount()
        {
            throw new NotImplementedException();
        }

        public override long GetDeviceMemorySize()
        {
            throw new NotImplementedException();
        }

        public override string GetDeviceName()
        {
            throw new NotImplementedException();
        }
    }

    internal class CUDADevice : ComputeDevice
    {
        public CUDADevice(ComputeDeviceDesc descriptor) : base(descriptor)
        {
        }

        public override List<List<NeuronData>> CalculateAccumulatedGradientForMinibatch(Network network, TrainingSuite suite, int trainingDataBegin, int trainingDataEnd)
        {
            throw new NotImplementedException();
        }

        public override float[] CalculateLayer(float[,] weightMx, float[] bias, float[] prevActivations, IActivationFunction sigmoidFunction)
        {
            throw new NotImplementedException();
        }

        public override void FlushWorkingCache()
        {
        }

        public override void Uninitialize()
        {
        }

        public static List<ComputeDeviceDesc> GetDevices()
        {
            throw new NotImplementedException();
            List<ComputeDeviceDesc> ret = new List<ComputeDeviceDesc>();
            return ret;
        }
    }
}
