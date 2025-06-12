

#ifdef VK_CONSTANTS_HOST
struct CalcSingleLayerPushConstantData
{
#define uint uint32_t
#elif defined VK_CONSTANTS_GLSL
layout(push_constant) uniform constants_
{
#endif

    uint weights_per_neuron;
    uint layer_neuron_count;
    uint activation_function;

#ifdef VK_CONSTANTS_HOST
};
#undef uint
#elif defined VK_CONSTANTS_GLSL
}
pc;
#endif