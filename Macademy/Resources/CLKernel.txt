///
/// OpenCL kernels implementing network calculations, and backpropagation
///

float ActivationFunction(int functionId, float x)
{
	switch(functionId)
	{
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

__kernel void calcSingleLayer(__global const float* weightMx, 
                              __global const float* biases, 
                              __global const float* prevActivation, 
                              __constant const int* config, 
                              __global float* output) 
{
    const int rowCount = config[0]; //number of neurons
    const int colCount = config[1]; //weights-per-neurons
    const int activationFunctionId = config[2]; 

    const int rowId = get_global_id(0);
 
    if (rowId >= rowCount)
        return;

    float acc = 0;
    for(int i = 0; i < colCount; ++i)
        acc += weightMx[(rowId * colCount) + i] * prevActivation[i];
    acc += biases[rowId];

    acc = ActivationFunction(activationFunctionId, acc);

    output[rowId] = acc;
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

int GetWeightAndBiasOffset(int layerId, __constant const int* layerSizes)
{
    int offset = 0;
    for(int i = 0; i < layerId; ++i){
        offset += layerSizes[i] * layerSizes[i+1] + layerSizes[i+1]; //prevWeights * neuroncCount (weights) + neuronCount (biases)
    }
    return offset;
}

int GetActivationLayerOffset(int layerId, __constant const int* layerSizes)
{
    int offset = 0;
    for(int i = 0; i < layerId; ++i){
        offset += layerSizes[i+1]; //neuron count of layer
    }
    return offset;
}

__kernel void trainingForwardPass(__constant const int* config, 
                                  __global float* activationsAndZValues, 
                                  __global const float* inputValues, 
                                  __global const float* weightsAndBiases) 
{
    const int layerId = config[0]; //The layer's index to be calculated
    //const int layerCount = config[1]; //number of layers in the network
    const int numTrainingSamples = config[2]; //number of layers in the network
    const int activationFunctionId = config[3]; //which activation function to use
    const int totalActivationCount = config[5]; //How many activations are there in total in the network
    const int weightsPerNeuron = config[8 + layerId]; //number of weights per neurons on the current layer
    const int neuronsInLayer = config[9 + layerId]; //number of neurons in the current layer
    __constant const int* layerNeuronCountBegin = config+8;

    const int neuronId = get_global_id(0);
    const int trainingSample = get_global_id(1);

    if (trainingSample >= numTrainingSamples || neuronId >= neuronsInLayer)
    {
        return;
    }

    const int activationTableOffset = totalActivationCount * trainingSample;

    const int weightOffset = GetWeightAndBiasOffset(layerId, layerNeuronCountBegin);
    const int biasOffset = weightOffset + neuronsInLayer * weightsPerNeuron;
    const int activationOffset = activationTableOffset + GetActivationLayerOffset(layerId, layerNeuronCountBegin);
    const int zValueOffset = activationOffset + (numTrainingSamples * totalActivationCount); //shift by table size
    __global const float* prevActivations = layerId == 0 ?
         (inputValues + weightsPerNeuron * trainingSample) :
         (activationsAndZValues + (activationTableOffset + GetActivationLayerOffset(layerId - 1, layerNeuronCountBegin)) );

    float acc = 0;
    for(int i = 0; i < weightsPerNeuron; ++i){
        acc += weightsAndBiases[weightOffset + (neuronId * weightsPerNeuron) + i] * prevActivations[i];
    }
    acc += weightsAndBiases[biasOffset + neuronId];

    activationsAndZValues[zValueOffset+neuronId] = acc;
    activationsAndZValues[activationOffset+neuronId] = ActivationFunction(activationFunctionId, acc);
}

__kernel void trainingBackwardPass(__constant const int* config,
                                   __global const float* activationsAndZValues, 
                                   __global float* delta_k_vector, 
                                   __global float* gradient, 
                                   __global const float* desiredOutputs, 
                                   __global const float* inputValues, 
                                   __global const float* weightsAndBiases) 
{
    const int layerId = config[0]; //The layer's index to be calculated
    const int layerCount = config[1]; //number of layers in the network
    const int numTrainingSamples = config[2]; //number of layers in the network
    const int activationFunctionId = config[3]; //which activation function to use
    const int costFunctionId = config[4]; //which cost function to use
    const int totalActivationCount = config[5]; //How many activations are there in total in the network
    //const int totalWeightAndBiasCount = config[6]; //How many weights and biases are in the network in all layers
    const int deltaKVectorStride = config[7]; //Table size of delta_k vector
    const int weightsPerNeuron = config[8 + layerId]; //number of weights per neurons on the current layer
    const int neuronsInLayer = config[9 + layerId]; //number of neurons in the current layer
    __constant const int* layerNeuronCountBegin = config+8;

    const int neuronId = get_global_id(0);
    const int trainingSample = get_global_id(1);

    if (trainingSample >= numTrainingSamples || neuronId >= neuronsInLayer)
    {
        return;
    }
    
    const int delta_k_read_offset = deltaKVectorStride * 2 * trainingSample + ((layerId % 2) * deltaKVectorStride);
    const int delta_k_write_offset = deltaKVectorStride * 2 * trainingSample + (((layerId+1) % 2) * deltaKVectorStride);

    const int activationTableOffset = totalActivationCount * trainingSample;
    const int activationOffset = activationTableOffset + GetActivationLayerOffset(layerId, layerNeuronCountBegin);
    const int zValueOffset = activationOffset + (numTrainingSamples * totalActivationCount); //shift by table size
    __global const float* prevActivations = layerId == 0 ?
         (inputValues + weightsPerNeuron * trainingSample) :
         (activationsAndZValues + (activationTableOffset + GetActivationLayerOffset(layerId - 1, layerNeuronCountBegin)) );
         
    const float zValue = activationsAndZValues[zValueOffset + neuronId];

    float delta_k;
    
    if ( layerId == (layerCount - 1) )
    {
        //Output layer
        const float activation = activationsAndZValues[activationOffset + neuronId];
        const float desiredOutput = desiredOutputs[trainingSample * neuronsInLayer + neuronId];
        delta_k = CostFunctionDelta(costFunctionId, activationFunctionId, zValue, activation, desiredOutput);
    }
    else 
    {
        //Hidden layer
        delta_k = 0;
        const int weightOffset = GetWeightAndBiasOffset(layerId + 1, layerNeuronCountBegin) + neuronId;
        const int nextLayerNeuronCount = config[10 + layerId];
        for(int i = 0; i < nextLayerNeuronCount; ++i)
            delta_k += delta_k_vector[delta_k_read_offset + i] * weightsAndBiases[weightOffset + (i * neuronsInLayer/*=weightsPerNeuron of the next layer*/)];
        delta_k *= ActivationFunctionPrime(activationFunctionId, zValue);
    }

    const int gradientBaseOffset = GetWeightAndBiasOffset(layerId, layerNeuronCountBegin) + neuronId * (weightsPerNeuron + 1);

    for(int i = 0; i < weightsPerNeuron; ++i)
        atomicAdd_g_f(gradient + gradientBaseOffset + i, delta_k * prevActivations[i]);
    atomicAdd_g_f(gradient + gradientBaseOffset + weightsPerNeuron, delta_k );

    if ( layerId != 0 )
        delta_k_vector[delta_k_write_offset + neuronId] = delta_k;
}
