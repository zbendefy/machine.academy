
#define VK_CONSTANTS_GLSL
#include "constants.h"

#define Activation_Sigmoid 0
#define Activation_ReLU 1
#define Activation_Tanh 2
#define Activation_LeakyReLU 3
#define Activation_Identity 4
#define Activation_Threshold 5
#define Activation_SoftPlus 6
#define Activation_ArcTan 7

float ActivationFunction(uint functionId, float x)
{
	switch (functionId) {
    case Activation_Sigmoid:
        return 1.0f / (1.0f + exp(-x));
    case Activation_ReLU:
        return x < 0.0f ? 0.0f : x;
    case Activation_Tanh:
        return 2.0f / (1.0f + exp(-2.0f * x)) - 1.0f;
    case Activation_Identity:
        return x;
    case Activation_Threshold:
        return x < 0 ? 0 : 1;
    case Activation_LeakyReLU:
        return x < 0.0f ? (0.01f * x) : x;
    case Activation_SoftPlus:
        return log(1 + exp(x));
    case Activation_ArcTan:
        return atan(x);
    default:
        return 0;
    }
}

float ActivationFunctionPrime(uint functionId, float x)
{
	switch (functionId) {
    case Activation_Sigmoid: {
        const float sigm = ActivationFunction(Activation_Sigmoid, x);
        return sigm * (1.0f - sigm);
    }
    case Activation_ReLU:
        return x < 0.0f ? 0.0f : 1.0f;
    case Activation_Tanh: {
        const float sigm = ActivationFunction(Activation_Tanh, x);
        return 1.0f - sigm * sigm;
    }
    case Activation_Identity:
        return 1.0f;
    case Activation_Threshold:
        return 0.0f;
    case Activation_LeakyReLU:
        return x < 0.0f ? 0.01f : 1.0f;
    case Activation_SoftPlus:
        return 1.0f / (1.0f + exp(-x));
    case Activation_ArcTan:
        return 1.0f / (x * x + 1);
    default:
        return 0;
    }
}

float CostFunctionDelta(uint costFunctionId, uint activationFunctionId, float z, float a, float desiredOutput)
{
	switch(costFunctionId)
	{
		case 0: //Mean-squared error cost function
			return (a - desiredOutput) * ActivationFunctionPrime(activationFunctionId, z);
		case 1: //Cross-entropy cost function
		default:
			return a - desiredOutput;
	}
}

uint GetLayerNeuronCountOffset(uint layerId)
{
    uint offset = 0;
    for(uint i = 0; i < layerId; ++i){
        offset += layer_config[2 + i * 2]; //neuron count of i-th layer 
    }
    return offset;
}

//Atomic addition function from: https://streamhpc.com/blog/2016-02-09/atomic-operations-for-floats-in-opencl-improved/
#if 0
void atomicAdd_g_f(volatile __global float *addr, float val)
{
    union {
        unsigned int u32;
        float        f32;
    } next, expected, current;
    current.f32    = *addr;
    do {
        expected.f32 = current.f32;
        next.f32     = expected.f32 + val;
        current.u32  = atomic_cmpxchg( (volatile __global unsigned int *)addr, expected.u32, next.u32);
    } while( current.u32 != expected.u32 );
}
#endif
