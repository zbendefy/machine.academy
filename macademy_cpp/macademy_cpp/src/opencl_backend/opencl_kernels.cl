constexpr const char* opencl_kernel_source = R"OPENCLSRC(
    
///
/// OpenCL kernels implementing network calculations, and backpropagation
///

float ActivationFunction(int functionId, float x)
{
	switch(functionId)
	{
        case 2: //ReLU
            return x < 0.0f ? 0.0f : x;
		case 1: //Sigmoid
			return 1.0f/(1.0f + exp(-x));
		case 0: //Passtrough
		default:
			return x;
	}
}

float ActivationFunctionPrime(int functionId, float x)
{
	switch(functionId)
	{
        case 2: //ReLU
            return x < 0.0f ? 0.0f : 1.0f;
		case 1: //Sigmoid
        {
			const float sigm = 1.0f/(1.0f + exp(-x));
            return sigm * (1.0f - sigm);
        }
		case 0: //Passtrough
		default:
			return 0;
	}
}

float CostFunctionDelta(int costFunctionId, int activationFunctionId, float z, float a, float desiredOutput)
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

__kernel void calcSingleLayer(__global const float* weights_biases,
                              __constant const uint* config, 
                              __global const float* input,
                              __global float* output) 
{
    const uint layer_neuron_count = config[0]; //number of neurons
    const uint weights_per_neuron = config[1]; //neurons in the prev layer
    const uint activationFunctionId = config[2]; 
    const uint weights_layer_offset = config[3]; //The offset where the weights and biases start

    const uint layer_neuron_id = get_global_id(0);
 
    if (layer_neuron_id >= layer_neuron_count)
        return;

    const uint neuron_data_size = weights_per_neuron + 1; //weights in prev layer + 1 bias

    __global const float* neuron_weights_biases = weights_biases + weights_layer_offset + layer_neuron_id * neuron_data_size;

    float acc = 0;
    for(int i = 0; i < weights_per_neuron; ++i)
    {
        acc += neuron_weights_biases[i] * input[i];
    }
    acc += neuron_weights_biases[weights_per_neuron]; //bias

    output[layer_neuron_id] = ActivationFunction(activationFunctionId, acc);
}

//Atomic addition function from: https://streamhpc.com/blog/2016-02-09/atomic-operations-for-floats-in-opencl-improved/
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


)OPENCLSRC";