
///
/// Vulkan kernels implementing network calculations, and backpropagation
///

enum ActivationFunction
{
#define Activation_Sigmoid 0
#define Activation_ReLU 1
#define Activation_Tanh 2
#define Activation_LeakyReLU 3
#define Activation_Identity 4
#define Activation_Threshold 5
#define Activation_SoftPlus 6
#define Activation_ArcTan 7
};

layout(std140, binding = 0) readonly buffer weights_biases_buf {
   float weights_biases[];
};

layout(std140, binding = 1) readonly buffer layer_config_buf {
   uint layer_config[];
};

layout(std140, binding = 2) buffer activationsAndZValues_buf {
   float activationsAndZValues[];
};

layout(std140, binding = 3) readonly buffer inputValues_buf {
   float inputValues[];
};

layout(std140, binding = 4) buffer delta_k_vector_buf {
   float delta_k_vector[];
};

layout(std140, binding = 5) buffer gradient_buf {
   float gradient[];
};

layout(std140, binding = 6) buffer desiredOutputs_buf {
   float desiredOutputs[];
};

layout( push_constant ) uniform constants
{
	uint layer_id;
    uint layer_count;
    uint weights_layer_offset; //ulong?
    uint numTrainingSamples;
    uint totalActivationCount;
    uint costFunctionId;
    uint largest_layer_neuron_count;
    uint layer_weights_offset; //ulong?
    float regularization_term_1;
    float regularization_term_2;
    float normalized_learning_rate;
} PushConstants;

int GetLayerNeuronCountOffset(int layerId, __constant const uint* layer_config)
{
    int offset = 0;
    for(int i = 0; i < layerId; ++i){
        offset += layer_config[2 + i * 2]; //neuron count of i-th layer 
    }
    return offset;
}


float ActivationFunction(int functionId, float x)
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

float ActivationFunctionPrime(int functionId, float x)
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

//Kernels
void evaluateLayerBatched(__global const float* weights_biases,
                                   __constant const uint* layer_config,
                                   __global const float* input_buffer,
                                   __global float* output_buffer,
                                   const uint layer_id,
                                   const ulong weights_layer_offset)
{
    const uint layer_neuron_count = layer_config[2 + layer_id * 2]; //number of neurons
    const uint weights_per_neuron = layer_config[layer_id*2]; //neurons in the prev layer
    const uint activationFunctionId = layer_config[3 + layer_id * 2]; 

    const uint layer_neuron_id = get_global_id(0);
    const uint batch_id = get_global_id(1);

    if (layer_neuron_id >= layer_neuron_count)
        return;

    __global const float* input = input_buffer + batch_id * weights_per_neuron;
    __global float* output = output_buffer + batch_id * layer_neuron_count;

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


void trainingForwardPass()
{
    const uint layer_neuron_count = layer_config[2 + layer_id * 2]; //number of neurons
    const uint weights_per_neuron = layer_config[layer_id*2]; //neurons in the prev layer
    const uint activationFunctionId = layer_config[3 + layer_id * 2]; 

    const uint layer_neuron_id = get_global_id(0);
    const uint trainingSampleId = get_global_id(1);

    if (trainingSampleId >= numTrainingSamples || layer_neuron_id >= layer_neuron_count)
    {
        return;
    }

    const uint neuron_data_size = weights_per_neuron + 1; //weights in prev layer + 1 bias

    __global const float* neuron_weights_biases = weights_biases + weights_layer_offset + layer_neuron_id * neuron_data_size;

    const int training_sample_activation_offset = totalActivationCount * trainingSampleId;

    const uint input_layer_neuron_count = layer_config[0];
    __global const float* prevActivations = layer_id == 0 ?
         (inputValues + input_layer_neuron_count * trainingSampleId) :
         (activationsAndZValues + (training_sample_activation_offset + GetLayerNeuronCountOffset(layer_id - 1, layer_config)) );
    
    //Calculate ZValues for layer
    float acc = 0;
    for(int i = 0; i < weights_per_neuron; ++i)
    {
        acc += neuron_weights_biases[i] * prevActivations[i];
    }
    acc += neuron_weights_biases[weights_per_neuron]; //bias

    //Store ZValues and the result of the activation function
    const int layer_activation_offset = training_sample_activation_offset + GetLayerNeuronCountOffset(layer_id, layer_config);
    const int layer_zvalue_offset = layer_activation_offset + (numTrainingSamples * totalActivationCount); //zvalues are stored after activations, so shift by the number of total activations
    activationsAndZValues[layer_zvalue_offset+layer_neuron_id] = acc;
    activationsAndZValues[layer_activation_offset+layer_neuron_id] = ActivationFunction(activationFunctionId, acc);
}

void trainingBackwardPass()
{
    const uint layer_neuron_count = layer_config[2 + layer_id * 2]; //number of neurons
    const uint weights_per_neuron = layer_config[layer_id*2]; //neurons in the prev layer
    const uint activationFunctionId = layer_config[3 + layer_id * 2]; 
    const uint deltaKVectorStride = largest_layer_neuron_count; //Table size of delta_k vector


    //__constant const int* layerNeuronCountBegin = config+8;

    const int layer_neuron_id = get_global_id(0);
    const int trainingSampleId = get_global_id(1);

    if (trainingSampleId >= numTrainingSamples || layer_neuron_id >= layer_neuron_count)
    {
        return;
    }
    
    const int delta_k_read_offset = deltaKVectorStride * 2 * trainingSampleId + ((layer_id % 2) * deltaKVectorStride);
    const int delta_k_write_offset = deltaKVectorStride * 2 * trainingSampleId + (((layer_id+1) % 2) * deltaKVectorStride);

    const int training_sample_activation_offset = totalActivationCount * trainingSampleId;
    const int current_layer_activation_offset = training_sample_activation_offset + GetLayerNeuronCountOffset(layer_id, layer_config);
    const int current_layer_z_values_offset = current_layer_activation_offset + (numTrainingSamples * totalActivationCount); //shift by table size

    const uint input_layer_neuron_count = layer_config[0];
    __global const float* prevActivations = layer_id == 0 ?
         (inputValues + input_layer_neuron_count * trainingSampleId) :
         (activationsAndZValues + (training_sample_activation_offset + GetLayerNeuronCountOffset(layer_id - 1, layer_config)) );
         
    const float zValue = activationsAndZValues[current_layer_z_values_offset + layer_neuron_id];

    float delta_k;
    
    if ( layer_id == (layer_count - 1) )
    {
        //Output layer
        const float activation = activationsAndZValues[current_layer_activation_offset + layer_neuron_id];
        const float desiredOutput = desiredOutputs[trainingSampleId * layer_neuron_count + layer_neuron_id];
        delta_k = CostFunctionDelta(costFunctionId, activationFunctionId, zValue, activation, desiredOutput);
    }
    else 
    {
        //Hidden layer
        delta_k = 0;
        const uint next_layer_weights_offset = layer_weights_offset + (layer_neuron_count * (weights_per_neuron + 1));
        const uint next_layer_weight_offset_for_processed_neuron = next_layer_weights_offset + layer_neuron_id;
        const uint next_layer_neuron_count = layer_config[2 + (layer_id+1) * 2]; //number of neurons
        const uint next_layer_neuron_data_size = layer_neuron_count + 1; //weights + bias
        for(uint i = 0; i < next_layer_neuron_count; ++i)
        {
            delta_k += delta_k_vector[delta_k_read_offset + i] * weightsAndBiases[next_layer_weight_offset_for_processed_neuron + (i * next_layer_neuron_data_size)];
        }
        delta_k *= ActivationFunctionPrime(activationFunctionId, zValue);
    }

    const uint gradientBaseOffset = layer_weights_offset + layer_neuron_id * (weights_per_neuron + 1);

    for(int i = 0; i < weights_per_neuron; ++i)
    {
        atomicAdd_g_f(gradient + gradientBaseOffset + i, delta_k * prevActivations[i]);
    }
    atomicAdd_g_f(gradient + gradientBaseOffset + weights_per_neuron, delta_k );

    if ( layer_id != 0 )
    {
        delta_k_vector[delta_k_write_offset + layer_neuron_id] = delta_k;
    }
}

void trainingApplyGradient()
{
    const uint layer_neuron_count = layer_config[2 + layer_id * 2]; //number of neurons
    const uint weights_per_neuron = layer_config[layer_id*2]; //neurons in the prev layer

    const uint layer_neuron_id = get_global_id(0);
 
    if (layer_neuron_id >= layer_neuron_count)
        return;

    const bool applyRegularizationTerm2 = regularization_term_2 != 0.0;

    __global float* neuron_weight_data = weights_biases + weights_layer_offset + layer_neuron_id * (weights_per_neuron + 1);
    __global float* neuron_gradient_data = gradient + weights_layer_offset + layer_neuron_id * (weights_per_neuron + 1);

    for (size_t j = 0; j < weights_per_neuron; ++j) {
        float weight = neuron_weight_data[j];
        weight = regularization_term_1 * weight - neuron_gradient_data[j] * normalized_learning_rate;
        if (applyRegularizationTerm2) {
            weight -= regularization_term_2 * sign(weight);
        }
        neuron_weight_data[j] = weight;
    }
    neuron_weight_data[weights_per_neuron] -= neuron_gradient_data[weights_per_neuron] * normalized_learning_rate; //bias
}
