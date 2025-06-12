#version 460
///
/// Vulkan kernels implementing network calculations, and backpropagation
///

#define VK_CONSTANTS_GLSL
#include "kernel_calc_single_layer_constants.h"

layout(std430, binding = 0) readonly buffer weights_biases_buf {
   float weights_biases[];
};

layout(std430, binding = 1) readonly buffer inputValues_buf {
   float input_buffer[];
};

layout(std430, binding = 2) writeonly buffer outputValues_buf {
   float output_buffer[];
};

#include "common.glsl"

layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

void main()
{
    const uint layer_neuron_id = gl_GlobalInvocationID.x;

    if (layer_neuron_id >= pc.layer_neuron_count)
        return;

    const uint neuron_data_size = pc.weights_per_neuron + 1; //weights in prev layer + 1 bias

    const uint neuron_weights_biases_begin_idx = layer_neuron_id * neuron_data_size;

    float acc = 0;
    for(uint i = 0; i < pc.weights_per_neuron; ++i)
    {
        acc += weights_biases[neuron_weights_biases_begin_idx + i] * input_buffer[i];
    }
    acc += weights_biases[neuron_weights_biases_begin_idx + pc.weights_per_neuron]; //bias

    output_buffer[layer_neuron_id] = ActivationFunction(pc.activation_function, acc);
}
