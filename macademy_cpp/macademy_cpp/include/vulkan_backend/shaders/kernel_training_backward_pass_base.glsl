#ifdef USE_SOFTWARE_ATOMIC_ADD

//Slower software path
#define GRADIENT_T uint
#define ATOMIC_ADD(__float_data, __add) { \
   uint next; \
   uint expected; \
   uint current = __float_data; \
   do { \
        expected = current; \
        next = floatBitsToUint(uintBitsToFloat(expected) + __add); \
        current = atomicCompSwap(__float_data, expected, next); \
    } while(current != expected); \
}

#else

//Faster hardware path
#extension GL_EXT_shader_atomic_float : enable
#define GRADIENT_T float
#define ATOMIC_ADD(a,b) atomicAdd(a,b)

#endif

#define VK_CONSTANTS_GLSL
#include "kernel_training_backward_pass_constants.h"

///
/// Vulkan kernels implementing network calculations, and backpropagation
///
layout(std430, binding = 0) readonly buffer next_layer_data_buf {
   float next_layer_data[];
};

layout(std430, binding = 1) readonly buffer prev_activations_buf {
   float prev_activations[];
};

layout(std430, binding = 2) readonly buffer layer_activations_buf {
   float layer_activations[];
};

layout(std430, binding = 3) readonly buffer layer_zvalues_buf {
   float layer_zvalues[];
};

layout(std430, binding = 4) buffer delta_k_vector_write_buf {
   float delta_k_vector_write[];
};

layout(std430, binding = 5) readonly buffer delta_k_vector_read_buf {
   float delta_k_vector_read[];
};

layout(std430, binding = 6) buffer current_layer_gradient_buf {
   GRADIENT_T current_layer_gradient[];
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
    const uint prev_layer_offset = prev_layer_neuron_count * trainingSampleId;
    const uint layer_offset = pc.layer_neuron_count * trainingSampleId;
    const uint next_layer_offset = pc.next_layer_neuron_count * trainingSampleId;
    const uint delta_k_read_offset = next_layer_offset;
    const uint delta_k_write_offset = layer_offset;

    const uint prev_activations_offset = prev_layer_offset;

    const float zValue = layer_zvalues[layer_offset + layer_neuron_id];

    float delta_k;
    
    if ( pc.is_output_layer != 0u )
    {
        //Output layer
        const float activation = layer_activations[layer_offset + layer_neuron_id];
        const float desiredOutput = next_layer_data[layer_offset + layer_neuron_id];
        delta_k = CostFunctionDelta(pc.cost_function, pc.activation_function, zValue, activation, desiredOutput);
    }
    else 
    {
        //Hidden layer
        delta_k = 0;
        const uint next_layer_neuron_data_size = pc.layer_neuron_count + 1;                   // weights + bias
        for(uint i = 0; i < pc.next_layer_neuron_count; ++i)
        {
            delta_k += delta_k_vector_read[delta_k_read_offset + i] * next_layer_data[layer_neuron_id + i * next_layer_neuron_data_size];
        }
        delta_k *= ActivationFunctionPrime(pc.activation_function, zValue);
    }

    const uint gradientBaseOffset = layer_neuron_id * (pc.weights_per_neuron + 1);

    for(uint i = 0; i < pc.weights_per_neuron; ++i)
    {
       ATOMIC_ADD(current_layer_gradient[gradientBaseOffset + i], delta_k * prev_activations[prev_activations_offset + i]);
    }
    ATOMIC_ADD(current_layer_gradient[gradientBaseOffset + pc.weights_per_neuron], delta_k );

   delta_k_vector_write[delta_k_write_offset + layer_neuron_id] = delta_k;
}
