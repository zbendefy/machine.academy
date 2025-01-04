#version 460
///
/// Vulkan kernels implementing network calculations, and backpropagation
///
layout(std430, binding = 0) buffer weights_biases_buf {
   float weights_biases[];
};

layout(std430, binding = 1) readonly buffer gradient_buf {
   float gradient[];
};

layout(std430, binding = 2) readonly buffer layer_config_buf {
   uint layer_config[];
};

#include "common.glsl"

layout (local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

void main()
{
    const uint layer_neuron_count = layer_config[2 + pc.layer_id * 2]; //number of neurons
    const uint weights_per_neuron = layer_config[pc.layer_id*2]; //neurons in the prev layer

    const uint layer_neuron_id = gl_GlobalInvocationID.x;
 
    if (layer_neuron_id >= layer_neuron_count)
        return;

    const bool applyRegularizationTerm2 = pc.regularization_term_2 != 0.0;

   const uint neuron_offset_idx = pc.weights_layer_offset + layer_neuron_id * (weights_per_neuron + 1);

    for (uint j = 0; j < weights_per_neuron; ++j) {
        float weight = weights_biases[neuron_offset_idx + j];
        weight = pc.regularization_term_1 * weight - gradient[neuron_offset_idx + j] * pc.normalized_learning_rate;
        if (applyRegularizationTerm2) {
            weight -= pc.regularization_term_2 * sign(weight);
        }
        weights_biases[neuron_offset_idx + j] = weight;
    }
    weights_biases[neuron_offset_idx + weights_per_neuron] -= gradient[neuron_offset_idx + weights_per_neuron] * pc.normalized_learning_rate; //bias

}
