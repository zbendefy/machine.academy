

#ifdef VK_CONSTANTS_HOST
struct TrainingForwardPassPushConstantData
{
#define uint uint32_t
#elif defined VK_CONSTANTS_GLSL
layout(push_constant) uniform constants_
{
#endif

    uint current_layer_id;
    uint current_layer_weights_offset; // ulong?
    uint numTrainingSamples;
    uint totalActivationCount;

#ifdef VK_CONSTANTS_HOST
};
#undef uint
#elif defined VK_CONSTANTS_GLSL
}
pc;
#endif