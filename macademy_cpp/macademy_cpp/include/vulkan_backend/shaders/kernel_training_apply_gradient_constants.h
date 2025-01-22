

#ifdef VK_CONSTANTS_HOST
struct ApplyGradientPushConstantData
{
#define uint uint32_t
#elif defined VK_CONSTANTS_GLSL
layout(push_constant) uniform constants_
{
#endif

    uint current_layer_id;
    uint current_layer_weights_offset; // ulong?
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