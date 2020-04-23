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
        protected ComputeDeviceDesc descriptor;

        protected ComputeDevice(ComputeDeviceDesc descriptor)
        {
            this.descriptor = descriptor;
        }

        public string GetDeviceAccessMode() { return descriptor.GetDeviceAccessType(); }

        public  long GetDeviceMemorySize() { return descriptor.GetDeviceMemorySize(); }

        public  int GetDeviceCoreCount() { return descriptor.GetDeviceCoreCount(); }

        public  string GetName() { return descriptor.GetDeviceName(); }

        public abstract unsafe float[] CalculateLayer(float[,] weightMx, float[] bias, float[] prevActivations, IActivationFunction sigmoidFunction);

        public abstract unsafe List<List<NeuronData>> CalculateAccumulatedGradientForMinibatch(Network network, TrainingSuite suite, int trainingDataBegin, int trainingDataEnd);

        public abstract void FlushWorkingCache();

        public abstract void Uninitialize();

        public override String ToString() { return GetName(); }

    }
}
