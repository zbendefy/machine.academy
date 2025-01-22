#version 460
#extension GL_EXT_shader_atomic_float : enable

#define VK_CONSTANTS_GLSL
#include "kernel_training_backward_pass_constants.h"

///
/// Vulkan kernels implementing network calculations, and backpropagation
///
layout(std430, binding = 0) readonly buffer weights_biases_buf {
   float weightsAndBiases[];
};

layout(std430, binding = 1) readonly buffer layer_config_buf {
   uint layer_config[];
};

layout(std430, binding = 2) readonly buffer activations_zvalues_buffer_buf {
   float activationsAndZValues[];
};

layout(std430, binding = 3) readonly buffer input_buf {
   float inputValues[];
};

layout(std430, binding = 4) buffer delta_k_vector_buf {
   float delta_k_vector[];
};

layout(std430, binding = 5) buffer gradient_buf {
   float gradient[];
};

layout(std430, binding = 6) readonly buffer desiredOutputs_buf {
   float desiredOutputs[];
};

#include "common.glsl"

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z = 1) in;

void main()
{
    const uint layer_neuron_count = layer_config[2 + pc.current_layer_id * 2]; //number of neurons
    const uint weights_per_neuron = layer_config[pc.current_layer_id*2]; //neurons in the prev layer
    const uint activationFunctionId = layer_config[3 + pc.current_layer_id * 2]; 
    const uint deltaKVectorStride = pc.largest_layer_neuron_count; //Table size of delta_k vector

    const uint layer_neuron_id = gl_GlobalInvocationID.x;
    const uint trainingSampleId = gl_GlobalInvocationID.y;

    if (trainingSampleId >= pc.numTrainingSamples || layer_neuron_id >= layer_neuron_count)
    {
        return;
    }
    
    const bool is_first_layer = pc.current_layer_id == 0;

    const uint delta_k_read_offset = deltaKVectorStride * 2 * trainingSampleId + ((pc.current_layer_id % 2) * deltaKVectorStride);
    const uint delta_k_write_offset = deltaKVectorStride * 2 * trainingSampleId + (((pc.current_layer_id+1) % 2) * deltaKVectorStride);

    const uint training_sample_activation_offset = pc.totalActivationCount * trainingSampleId;
    const uint current_layer_activation_offset = training_sample_activation_offset + GetLayerNeuronCountOffset(pc.current_layer_id);
    const uint current_layer_z_values_offset = current_layer_activation_offset + (pc.layer_count * pc.totalActivationCount); //shift by table size

    const uint input_layer_neuron_count = layer_config[0];

    const uint prev_layer_input_values_idx = input_layer_neuron_count * trainingSampleId;
    const uint prev_layer_activations_idx = is_first_layer ? 0 : (training_sample_activation_offset + GetLayerNeuronCountOffset(pc.current_layer_id - 1));

    const float zValue = activationsAndZValues[current_layer_z_values_offset + layer_neuron_id];

    float delta_k;
    
    if ( pc.current_layer_id == (pc.layer_count - 1) )
    {
        //Output layer
        const float activation = activationsAndZValues[current_layer_activation_offset + layer_neuron_id];
        const float desiredOutput = desiredOutputs[trainingSampleId * layer_neuron_count + layer_neuron_id];
        delta_k = CostFunctionDelta(pc.costFunctionId, activationFunctionId, zValue, activation, desiredOutput);
    }
    else 
    {
        //Hidden layer
        delta_k = 0;
        const uint next_layer_weights_offset = pc.current_layer_weights_offset + (layer_neuron_count * (weights_per_neuron + 1));
        const uint next_layer_weight_offset_for_processed_neuron = next_layer_weights_offset + layer_neuron_id;
        const uint next_layer_neuron_count = layer_config[2 + (pc.current_layer_id+1) * 2]; //number of neurons
        const uint next_layer_neuron_data_size = layer_neuron_count + 1; //weights + bias
        for(uint i = 0; i < next_layer_neuron_count; ++i)
        {
            delta_k += delta_k_vector[delta_k_read_offset + i] * weightsAndBiases[next_layer_weight_offset_for_processed_neuron + (i * next_layer_neuron_data_size)];
        }
        delta_k *= ActivationFunctionPrime(activationFunctionId, zValue);
    }

    const uint gradientBaseOffset = pc.current_layer_weights_offset + layer_neuron_id * (weights_per_neuron + 1);

    for(uint i = 0; i < weights_per_neuron; ++i)
    {
       const float prev_activation = is_first_layer ? inputValues[prev_layer_input_values_idx + i] : activationsAndZValues[prev_layer_activations_idx + i];
       atomicAdd(gradient[gradientBaseOffset + i], delta_k * prev_activation);
    }
    atomicAdd(gradient[gradientBaseOffset + weights_per_neuron], delta_k );

    if ( pc.current_layer_id != 0 )
    {
        delta_k_vector[delta_k_write_offset + layer_neuron_id] = delta_k;
    }
}
