using System;
using System.Collections.Generic;
using System.Text;

namespace Macademy
{
    public abstract class ComputeDeviceDesc
    {
        public abstract string GetDeviceAccessType();
        public abstract string GetDeviceName();
        public abstract int GetDeviceCoreCount();
        public abstract long GetDeviceMemorySize();
        public abstract ComputeDevice CreateDevice();
    }

    public abstract class ComputeDevice
    {
        public abstract string GetDeviceAccessMode();

        public abstract int GetDeviceCoreCount();

        public abstract string GetName();

        public abstract unsafe float[] CalculateLayer(float[,] weightMx, float[] bias, float[] prevActivations, IActivationFunction sigmoidFunction);

        public abstract unsafe List<List<NeuronData>> CalculateAccumulatedGradientForMinibatch(Network network, TrainingSuite suite, int trainingDataBegin, int trainingDataEnd);

        public abstract void FlushWorkingCache();

        public abstract void Uninitialize();

        public override String ToString() { return GetName(); }

    }
}
