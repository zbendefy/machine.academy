#version 460
///
/// Vulkan kernels implementing network calculations, and backpropagation
///

#define VK_CONSTANTS_GLSL
#include "kernel_training_forward_pass_constants.h"

layout(std430, binding = 0) readonly buffer weights_biases_buf {
   float weights_biases[];
};

layout(std430, binding = 1) readonly buffer prev_activations_buf {
   float prev_activations[];
};

layout(std430, binding = 2) buffer activations_buf {
   float activations[];
};

layout(std430, binding = 3) buffer zvalues_buf {
   float zvalues[];
};

#include "common.glsl"

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z = 1) in;

void main()
{
    const uint layer_neuron_id = gl_GlobalInvocationID.x;
    const uint trainingSampleId = gl_GlobalInvocationID.y;

    if (trainingSampleId >= pc.num_training_samples || layer_neuron_id >= pc.layer_neuron_count)
    {
        return;
    }

   const uint prev_layer_neuron_count = pc.weights_per_neuron;
   const uint neuron_data_size = pc.weights_per_neuron + 1;
   const uint prev_layer_offset = prev_layer_neuron_count * trainingSampleId;
   const uint layer_offset = pc.layer_neuron_count * trainingSampleId;
   const uint neuron_weights_biases_base_offset = layer_neuron_id * neuron_data_size;

   // Calculate ZValues for layer
   float acc = 0;
   for (uint i = 0; i < pc.weights_per_neuron; ++i) {
      acc += weights_biases[neuron_weights_biases_base_offset + i] * prev_activations[prev_layer_offset + i];
   }
   acc += weights_biases[neuron_weights_biases_base_offset + pc.weights_per_neuron]; // bias

   // Store ZValues and the result of the activation function
   zvalues[layer_offset + layer_neuron_id] = acc;
   activations[layer_offset + layer_neuron_id] = ActivationFunction(pc.activation_function, acc);
}
