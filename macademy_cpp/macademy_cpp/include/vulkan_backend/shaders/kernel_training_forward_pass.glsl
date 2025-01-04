#version 460
///
/// Vulkan kernels implementing network calculations, and backpropagation
///
layout(std430, binding = 0) readonly buffer weights_biases_buf {
   float weights_biases[];
};

layout(std430, binding = 1) readonly buffer layer_config_buf {
   uint layer_config[];
};

layout(std430, binding = 2) buffer activations_zvalues_buffer_buf {
   float activationsAndZValues[];
};

layout(std430, binding = 3) readonly buffer input_buf {
   float inputValues[];
};

#include "common.glsl"

layout (local_size_x = 64, local_size_y = 1, local_size_z = 1) in;


__kernel void trainingForwardPass(__global const float* weights_biases,
                                  __constant const uint* layer_config,
                                  __global float* activationsAndZValues, //Stores activations for all layers, and then zvalues for all layers
                                  __global const float* inputValues,
                                  const uint layer_id,
                                  const ulong weights_layer_offset,
                                  const uint numTrainingSamples,
                                  const uint totalActivationCount
                                  )
{
    }

void main()
{
    const uint layer_neuron_count = layer_config[2 + layer_id * 2]; //number of neurons
    const uint weights_per_neuron = layer_config[layer_id*2]; //neurons in the prev layer
    const uint activationFunctionId = layer_config[3 + layer_id * 2]; 

    const uint layer_neuron_id = gl_GlobalInvocationID.x;
    const uint trainingSampleId = gl_GlobalInvocationID.y;

    if (trainingSampleId >= numTrainingSamples || layer_neuron_id >= layer_neuron_count)
    {
        return;
    }

    const uint neuron_data_size = weights_per_neuron + 1; //weights in prev layer + 1 bias

    //__global const float* neuron_weights_biases = weights_biases + weights_layer_offset + layer_neuron_id * neuron_data_size;
   const uint neuron_weights_biases_idx = weights_layer_offset + layer_neuron_id * neuron_data_size;

    const int training_sample_activation_offset = totalActivationCount * trainingSampleId;

    const uint input_layer_neuron_count = layer_config[0];
    /*__global const float* prevActivations = layer_id == 0 ?
         (inputValues + input_layer_neuron_count * trainingSampleId) :
         (activationsAndZValues + (training_sample_activation_offset + GetLayerNeuronCountOffset(layer_id - 1, layer_config)) );
    */
    const uint prevActivations_idx = ...;

    //Calculate ZValues for layer
    float acc = 0;
    for(int i = 0; i < weights_per_neuron; ++i)
    {
        acc += weights_biases[i + neuron_weights_biases_idx] * prevActivations[i];
    }
    acc += weights_biases[weights_per_neuron + neuron_weights_biases_idx]; //bias

    //Store ZValues and the result of the activation function
    const int layer_activation_offset = training_sample_activation_offset + GetLayerNeuronCountOffset(layer_id, layer_config);
    const int layer_zvalue_offset = layer_activation_offset + (numTrainingSamples * totalActivationCount); //zvalues are stored after activations, so shift by the number of total activations
    activationsAndZValues[layer_zvalue_offset+layer_neuron_id] = acc;
    activationsAndZValues[layer_activation_offset+layer_neuron_id] = ActivationFunction(activationFunctionId, acc);
}
