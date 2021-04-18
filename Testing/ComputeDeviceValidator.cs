using Macademy;
using System;
using System.Collections.Generic;
using System.Text;

namespace ModuleTests
{
    class ValidatorComputeDeviceDesc : ComputeDeviceDesc
    {
        public override ComputeDevice CreateDevice()
        {
            throw new NotImplementedException();
        }

        public override string GetDeviceAccessType()
        {
            throw new NotImplementedException();
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

    class ComputeDeviceValidator : ComputeDevice
    {
        ComputeDevice[] devices;

        public ComputeDeviceValidator(ComputeDevice[] devices)
            :base(new ValidatorComputeDeviceDesc())
        {
            this.devices = devices;
        }

        public override List<List<NeuronData>> CalculateAccumulatedGradientForMinibatch(Network network, TrainingSuite suite, int trainingDataBegin, int trainingDataEnd)
        {
            List<List<NeuronData>> ret = null;
            foreach (var device in devices)
            {
                var result = device.CalculateAccumulatedGradientForMinibatch(network, suite, trainingDataBegin, trainingDataEnd);
                if (ret != null)
                {
                    Utils.ValidateGradient(ret, result, 0.00001);
                }
                ret = result;
            }
            return ret;
        }


        public override void FlushWorkingCache()
        {
            foreach (var device in devices)
            {
                device.FlushWorkingCache();
            }
        }

        public override void Uninitialize()
        {
            foreach (var device in devices)
            {
                device.Uninitialize();
            }
        }

        public override float[] EvaluateNetwork(float[] input, Network network)
        {
            float[] ret = null;
            foreach (var device in devices)
            {
                var result = device.EvaluateNetwork(input, network);
                if (ret != null)
                {
                    Utils.ValidateFloatArray(ret, result);
                }
                ret = result;
            }
            return ret;
        }
    }
}
