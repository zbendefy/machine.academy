

#ifdef VK_CONSTANTS_HOST
struct TrainingForwardPassPushConstantData
{
#define uint uint32_t
#elif defined VK_CONSTANTS_GLSL
layout(push_constant) uniform constants_
{
#endif

    uint activation_function;
    uint layer_neuron_count;
    uint weights_per_neuron;
    uint num_training_samples;

#ifdef VK_CONSTANTS_HOST
};
#undef uint
#elif defined VK_CONSTANTS_GLSL
}
pc;
#endif