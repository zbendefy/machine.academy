constexpr const char* opencl_kernel_source = R"OPENCLSRC(///
/// OpenCL kernels implementing network calculations, and backpropagation
///

enum ActivationFunction
{
    Activation_Sigmoid,
    Activation_ReLU,
    Activation_Tanh,
    Activation_LeakyReLU,
    Activation_Identity,
    Activation_Threshold,
    Activation_SoftPlus,
    Activation_ArcTan,
};

float ActivationFunction(uint functionId, float x)
{
    switch (functionId) {
    case Activation_Sigmoid:
        return 1.0f / (1.0f + exp(-x));
    case Activation_ReLU:
        return x < 0.0f ? 0.0f : x;
    case Activation_Tanh:
        return 2.0f / (1.0f + exp(-2.0f * x)) - 1.0f;
    case Activation_Identity:
        return x;
    case Activation_Threshold:
        return x < 0 ? 0 : 1;
    case Activation_LeakyReLU:
        return x < 0.0f ? (0.01f * x) : x;
    case Activation_SoftPlus:
        return log(1 + exp(x));
    case Activation_ArcTan:
        return atan(x);
    default:
        return 0.0f;
    }
}

float ActivationFunctionPrime(uint functionId, float x)
{
    switch (functionId) {
    case Activation_Sigmoid: {
        const float sigm = ActivationFunction(Activation_Sigmoid, x);
        return sigm * (1.0f - sigm);
    }
    case Activation_ReLU:
        return x < 0.0f ? 0.0f : 1.0f;
    case Activation_Tanh: {
        const float sigm = ActivationFunction(Activation_Tanh, x);
        return 1.0f - sigm * sigm;
    }
    case Activation_Identity:
        return 1.0f;
    case Activation_Threshold:
        return 0.0f;
    case Activation_LeakyReLU:
        return x < 0.0f ? 0.01f : 1.0f;
    case Activation_SoftPlus:
        return 1.0f / (1.0f + exp(-x));
    case Activation_ArcTan:
        return 1.0f / (x * x + 1);
    default:
        return 0.0f;
    }
}

float CostFunctionDelta(uint costFunctionId, uint activationFunctionId, float z, float a, float desiredOutput)
{
    switch (costFunctionId) {
    case 0: // Mean-squared error cost function
        return (a - desiredOutput) * ActivationFunctionPrime(activationFunctionId, z);
    case 1: // Cross-entropy cost function
    default:
        return a - desiredOutput;
    }
}

__kernel void evaluateLayer(__global const float* weights_biases, __global const float* input_buffer, __global float* output_buffer,
                                   const uint activation_function, const uint layer_input_count, const uint layer_neuron_count)
{
    const uint weights_per_neuron = layer_input_count;     // neurons in the prev layer

    const uint layer_neuron_id = get_global_id(0);

    if (layer_neuron_id >= layer_neuron_count)
        return;

    const uint neuron_data_size = weights_per_neuron + 1; // weights in prev layer + 1 bias

    __global const float* neuron_weights_biases = weights_biases + layer_neuron_id * neuron_data_size;

    float acc = 0.0f;
    for (uint i = 0; i < weights_per_neuron; ++i) {
        acc += neuron_weights_biases[i] * input_buffer[i];
    }
    acc += neuron_weights_biases[weights_per_neuron]; // bias

    output_buffer[layer_neuron_id] = ActivationFunction(activation_function, acc);
}

// Atomic addition function from: https://streamhpc.com/blog/2016-02-09/atomic-operations-for-floats-in-opencl-improved/
void atomicAdd_g_f(volatile __global float* addr, float val)
{
    union
    {
        unsigned int u32;
        float f32;
    } next, expected, current;
    current.f32 = *addr;
    do {
        expected.f32 = current.f32;
        next.f32 = expected.f32 + val;
        current.u32 = atomic_cmpxchg((volatile __global unsigned int*)addr, expected.u32, next.u32);
    } while (current.u32 != expected.u32);
}

uint GetLayerNeuronCountOffset(uint layerId, __constant const uint* layer_config)
{
    uint offset = 0;
    for (uint i = 0; i < layerId; ++i) {
        offset += layer_config[2 + i * 2]; // neuron count of i-th layer
    }
    return offset;
}

__kernel void trainingForwardPass(__global const float* weights_biases,
                                  __global const float* prev_activations_base,
                                  __global float* activations,
                                  __global float* zvalues,
                                  const uint activation_function,
                                  const uint layer_neuron_count,
                                  const uint weights_per_neuron,
                                  const uint numTrainingSamples)
{
    const uint layer_neuron_id = get_global_id(0);
    const uint trainingSampleId = get_global_id(1);

    if (trainingSampleId >= numTrainingSamples || layer_neuron_id >= layer_neuron_count) {
        return;
    }

    const uint prev_layer_neuron_count = weights_per_neuron;
    const uint neuron_data_size = weights_per_neuron + 1; // weights in prev layer + 1 bias

    const int prev_layer_offset = prev_layer_neuron_count * trainingSampleId;
    const int layer_offset = layer_neuron_count * trainingSampleId;
    
    __global const float* neuron_weights_biases = weights_biases + layer_neuron_id * neuron_data_size;

    __global const float* prevActivations = prev_activations_base + prev_layer_offset;

    // Calculate ZValues for layer
    float acc = 0;
    for (uint i = 0; i < weights_per_neuron; ++i) {
        acc += neuron_weights_biases[i] * prevActivations[i];
    }
    acc += neuron_weights_biases[weights_per_neuron]; // bias

    // Store ZValues and the result of the activation function
    zvalues[layer_offset + layer_neuron_id] = acc;
    activations[layer_offset + layer_neuron_id] = ActivationFunction(activation_function, acc);
}

__kernel void trainingBackwardPass( __global const float* next_layer_weights,
                                    __global const float* prev_activations_base,
                                    __global const float* layer_activations,
                                    __global const float* layer_zvalues,
                                    __global float* delta_k_vector_write,
                                    __global const float* delta_k_vector_read,
                                    __global float* current_layer_gradient,
                                    __global const float* desired_output,
                                    const uint layer_neuron_count,
                                    const uint weights_per_neuron,
                                    const uint activation_function,
                                    const uint numTrainingSamples,
                                    const uint cost_function,
                                    const uint next_layer_neuron_count,
                                    const uint is_output_layer)
{
    const uint layer_neuron_id = get_global_id(0);
    const uint trainingSampleId = get_global_id(1);

    if (trainingSampleId >= numTrainingSamples || layer_neuron_id >= layer_neuron_count) {
        return;
    }

    const uint prev_layer_neuron_count = weights_per_neuron;
    const uint prev_layer_offset = prev_layer_neuron_count * trainingSampleId;
    const uint layer_offset = layer_neuron_count * trainingSampleId;
    const uint next_layer_offset = next_layer_neuron_count * trainingSampleId;
    const uint delta_k_read_offset = next_layer_offset;
    const uint delta_k_write_offset = layer_offset;

    __global const float* prev_activations = prev_activations_base + prev_layer_offset;

    const float zValue = layer_zvalues[layer_neuron_id + layer_offset];

    float delta_k;

    if (is_output_layer) {
        // Output layer
        const float activation = layer_activations[layer_neuron_id + layer_offset];
        const float desiredOutput = desired_output[layer_neuron_id + layer_offset];
        delta_k = CostFunctionDelta(cost_function, activation_function, zValue, activation, desiredOutput);
    } else {
        // Hidden layer
        delta_k = 0;
        const uint next_layer_neuron_data_size = layer_neuron_count + 1;                   // weights + bias
        for (uint i = 0; i < next_layer_neuron_count; ++i) {
            delta_k += delta_k_vector_read[delta_k_read_offset + i] * next_layer_weights[layer_neuron_id + i * next_layer_neuron_data_size];
        }
        delta_k *= ActivationFunctionPrime(activation_function, zValue);
    }

    const uint gradientBaseOffset = layer_neuron_id * (weights_per_neuron + 1);

    for (uint i = 0; i < weights_per_neuron; ++i) {
        __global float* v = (current_layer_gradient + gradientBaseOffset + i);
        v[0] += delta_k * prev_activations[i];
    }
    __global float* v = (current_layer_gradient + gradientBaseOffset + weights_per_neuron);
    v[0] += delta_k; // bias

    //TODOZ: if this is the input layer of the network, this write is unnecessary, as it won't be used. This write can be omitted
    delta_k_vector_write[delta_k_write_offset + layer_neuron_id] = delta_k;
}

__kernel void trainingApplyGradient(__global float* weights_biases,
                                    __global const float* gradient,
                                    const uint layer_neuron_count,
                                    const uint weights_per_neuron,
                                    const float regularization_term_1,
                                    const float regularization_term_2,
                                    const float normalized_learning_rate)
{
    const uint layer_neuron_id = get_global_id(0);

    if (layer_neuron_id >= layer_neuron_count)
        return;

    const bool applyRegularizationTerm2 = regularization_term_2 != 0.0f;

    __global float* neuron_weight_data = weights_biases + layer_neuron_id * (weights_per_neuron + 1);
    const __global float* neuron_gradient_data = gradient + layer_neuron_id * (weights_per_neuron + 1);

    for (size_t j = 0; j < weights_per_neuron; ++j) {
        float weight = neuron_weight_data[j];
        weight = regularization_term_1 * weight - neuron_gradient_data[j] * normalized_learning_rate;
        if (applyRegularizationTerm2) {
            weight -= regularization_term_2 * sign(weight);
        }
        neuron_weight_data[j] = weight;
    }
    neuron_weight_data[weights_per_neuron] -= neuron_gradient_data[weights_per_neuron] * normalized_learning_rate; // bias
}
)OPENCLSRC";