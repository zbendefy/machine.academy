

#ifdef VK_CONSTANTS_HOST
struct PushConstantData
{
#define uint uint32_t
#elif defined VK_CONSTANTS_GLSL
layout(push_constant) uniform constants_
{
#endif

    uint layer_id;
    uint layer_count;
    uint weights_layer_offset; // ulong?
    uint numTrainingSamples;
    uint totalActivationCount;
    uint costFunctionId;
    uint largest_layer_neuron_count;
    uint layer_weights_offset; // ulong?
    float regularization_term_1;
    float regularization_term_2;
    float normalized_learning_rate;

#ifdef VK_CONSTANTS_HOST
};
#undef uint
#elif defined VK_CONSTANTS_GLSL
}
pc;
#endif