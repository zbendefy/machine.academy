#version 460
///
/// Vulkan kernels implementing network calculations, and backpropagation
///

#define VK_CONSTANTS_GLSL
#include "kernel_training_apply_gradient_constants.h"

layout(std430, binding = 0) buffer weights_biases_buf {
   float weights_biases[];
};

layout(std430, binding = 1) readonly buffer gradient_buf {
   float gradient[];
};

#include "common.glsl"

layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

void main()
{
    const uint layer_neuron_id = gl_GlobalInvocationID.x;
 
    if (layer_neuron_id >= pc.layer_neuron_count)
        return;

    const bool applyRegularizationTerm2 = pc.regularization_term_2 != 0.0;

    const uint neuron_offset_idx = layer_neuron_id * (pc.weights_per_neuron + 1);

    for (uint j = 0; j < pc.weights_per_neuron; ++j) {
        float weight = weights_biases[neuron_offset_idx + j];
        weight = pc.regularization_term_1 * weight - gradient[neuron_offset_idx + j] * pc.normalized_learning_rate;
        if (applyRegularizationTerm2) {
            weight -= pc.regularization_term_2 * sign(weight);
        }
        weights_biases[neuron_offset_idx + j] = weight;
    }
    weights_biases[neuron_offset_idx + pc.weights_per_neuron] -= gradient[neuron_offset_idx + pc.weights_per_neuron] * pc.normalized_learning_rate; //bias
}
