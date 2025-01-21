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

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

void main()
{
    const uint layer_neuron_count = layer_config[2 + pc.layer_id * 2]; //number of neurons
    const uint weights_per_neuron = layer_config[pc.layer_id*2]; //neurons in the prev layer
    const uint activationFunctionId = layer_config[3 + pc.layer_id * 2]; 

    const uint layer_neuron_id = gl_GlobalInvocationID.x;
    const uint trainingSampleId = gl_GlobalInvocationID.y;

    if (trainingSampleId >= pc.numTrainingSamples || layer_neuron_id >= layer_neuron_count)
    {
        return;
    }

    const bool is_first_layer = pc.layer_id == 0;

    const uint neuron_data_size = weights_per_neuron + 1; //weights in prev layer + 1 bias

    const uint neuron_weights_biases_idx = pc.weights_layer_offset + layer_neuron_id * neuron_data_size;

    const uint training_sample_activation_offset = pc.totalActivationCount * trainingSampleId;

    const uint input_layer_neuron_count = layer_config[0];

    const uint prev_layer_input_values_idx = input_layer_neuron_count * trainingSampleId;
    const uint prev_layer_activations_idx = is_first_layer ? 0 : (training_sample_activation_offset + GetLayerNeuronCountOffset(pc.layer_id - 1));

    //Calculate ZValues for layer
    float acc = 0;
    for(uint i = 0; i < weights_per_neuron; ++i)
    {
        const float prev_activation = is_first_layer ? inputValues[prev_layer_input_values_idx + i] : activationsAndZValues[prev_layer_activations_idx + i];
        acc += weights_biases[i + neuron_weights_biases_idx] * prev_activation;
    }
    acc += weights_biases[weights_per_neuron + neuron_weights_biases_idx]; //bias

    //Store ZValues and the result of the activation function
    const uint layer_activation_offset = training_sample_activation_offset + GetLayerNeuronCountOffset(pc.layer_id);
    const uint layer_zvalue_offset = layer_activation_offset + (pc.numTrainingSamples * pc.totalActivationCount); //zvalues are stored after activations, so shift by the number of total activations
    activationsAndZValues[layer_zvalue_offset+layer_neuron_id] = acc;
    activationsAndZValues[layer_activation_offset+layer_neuron_id] = ActivationFunction(activationFunctionId, acc);
}
