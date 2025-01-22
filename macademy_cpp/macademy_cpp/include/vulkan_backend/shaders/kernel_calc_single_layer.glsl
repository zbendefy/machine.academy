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

layout(std430, binding = 2) readonly buffer inputValues_buf {
   float input_buffer[];
};

layout(std430, binding = 3) buffer outputValues_buf {
   float output_buffer[];
};

#include "common.glsl"

layout (local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

void main()
{
    const uint layer_neuron_count = layer_config[2 + pc.current_layer_id * 2]; //number of neurons
    const uint weights_per_neuron = layer_config[pc.current_layer_id*2]; //neurons in the prev layer
    const uint activationFunctionId = layer_config[3 + pc.current_layer_id * 2]; 

    const uint layer_neuron_id = gl_GlobalInvocationID.x;
    const uint batch_id = gl_GlobalInvocationID.y;

    if (layer_neuron_id >= layer_neuron_count)
        return;

    uint input_begin_idx = batch_id * weights_per_neuron;
    uint output_begin_idx = batch_id * layer_neuron_count;

    const uint neuron_data_size = weights_per_neuron + 1; //weights in prev layer + 1 bias

    uint neuron_weights_biases_begin_idx = pc.current_layer_weights_offset + layer_neuron_id * neuron_data_size;

    float acc = 0;
    for(uint i = 0; i < weights_per_neuron; ++i)
    {
        acc += weights_biases[neuron_weights_biases_begin_idx + i] * input_buffer[input_begin_idx + i];
    }
    acc += weights_biases[neuron_weights_biases_begin_idx + weights_per_neuron]; //bias

    output_buffer[output_begin_idx + layer_neuron_id] = ActivationFunction(activationFunctionId, acc);
}
