

#ifdef VK_CONSTANTS_HOST
struct TrainingBackwardPassPushConstantData
{
#define uint uint32_t
#elif defined VK_CONSTANTS_GLSL
layout(push_constant) uniform constants_
{
#endif

    uint layer_neuron_count;
    uint weights_per_neuron;
    uint activation_function;
    uint num_training_samples;
    uint cost_function;
    uint next_layer_neuron_count;
    uint is_output_layer;

#ifdef VK_CONSTANTS_HOST
};
#undef uint
#elif defined VK_CONSTANTS_GLSL
}
pc;
#endif