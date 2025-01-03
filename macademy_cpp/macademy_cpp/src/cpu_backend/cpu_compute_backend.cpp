#include "cpu_backend/cpu_compute_backend.h"
#include "network.h"
#include "common.h"
#include "utils.h"
#include "training_suite.h"
#include "hwinfo/hwinfo.h"
#include <execution>
#include <algorithm>

namespace macademy {
namespace {

int GetLayerNeuronCountOffset(int layerId, const uint32_t* layer_config)
{
    int offset = 0;
    for (int i = 0; i < layerId; ++i) {
        offset += layer_config[2 + i * 2]; // neuron count of i-th layer
    }
    return offset;
}

inline float CalculateActivationFunction(ActivationFunction func, float x)
{
    switch (func) {
    case ActivationFunction::Sigmoid:
        return 1.0f / (1.0f + expf(-x));
    case ActivationFunction::ReLU:
        return x < 0.0f ? 0.0f : x;
    case ActivationFunction::Tanh:
        return 2.0f / (1.0f + expf(-2.0f * x)) - 1.0f;
    case ActivationFunction::Identity:
        return x;
    case ActivationFunction::Threshold:
        return x < 0 ? 0 : 1;
    case ActivationFunction::LeakyReLU:
        return x < 0.0f ? (0.01f * x) : x;
    case ActivationFunction::SoftPlus:
        return logf(1 + exp(x));
    case ActivationFunction::ArcTan:
        return atanf(x);
    }

    throw std::runtime_error("Invalid activation function!");
}

inline float CalculateActivationFunctionPrime(ActivationFunction func, float x)
{
    switch (func) {
    case ActivationFunction::Sigmoid: {
        const float sigm = CalculateActivationFunction(ActivationFunction::Sigmoid, x);
        return sigm * (1.0f - sigm);
    }
    case ActivationFunction::ReLU:
        return x < 0.0f ? 0.0f : 1.0f;
    case ActivationFunction::Tanh: {
        const float sigm = CalculateActivationFunction(ActivationFunction::Tanh, x);
        return 1.0f - sigm * sigm;
    }
    case ActivationFunction::Identity:
        return 1.0f;
    case ActivationFunction::Threshold:
        return 0.0f;
    case ActivationFunction::LeakyReLU:
        return x < 0.0f ? 0.01f : 1.0f;
    case ActivationFunction::SoftPlus:
        return 1.0f / (1.0f + expf(-x));
    case ActivationFunction::ArcTan:
        return 1.0f / (x * x + 1);
    }

    throw std::runtime_error("Invalid activation function!");
}

inline float CalculateCostFunctionError(CostFunction cost_fnc, float result, float desired_output)
{
    switch (cost_fnc) {
    case CostFunction::MeanSquared: {
        float v = result - desired_output;
        return 0.5f * v * v;
    }
    case CostFunction::CrossEntropy_Sigmoid: {
        return -desired_output * logf(result) - (1.0f - desired_output) * logf(1.0f - result);
    }
    }
    throw std::runtime_error("Invalid cost function!");
}

inline float CalculateCostFunctionDelta(CostFunction cost_fnc, ActivationFunction activation_function, float z, float a, float desired_output)
{
    switch (cost_fnc) {
    case CostFunction::MeanSquared: {
        return (a - desired_output) * CalculateActivationFunctionPrime(activation_function, z);
    }
    case CostFunction::CrossEntropy_Sigmoid: {
        return a - desired_output;
    }
    }
    throw std::runtime_error("Invalid cost function!");
}

} // namespace

#if 0
void CPUComputeDevice::Train(NetworkResourceHandle& network_handle, const TrainingSuite& training_suite, uint32_t trainingDataBegin, uint32_t trainingDataEnd) const
{
    ASSERT(trainingDataBegin < trainingDataEnd);

    auto& network = *network_handle.m_network;
    const auto network_layout = network.GetLayerConfig();

    const auto accumulated_gradient = CalculateAccumulatedGradientForBatch(network_handle, training_suite, trainingDataBegin, trainingDataEnd);

    // Calculate regularization terms based on the training configuration
    float regularizationTerm1 = 1.0f;
    float regularizationTerm2Base = 0.0f;
    if (training_suite.m_regularization == Regularization::L2) {
        regularizationTerm1 = 1.0f - training_suite.m_learning_rate * (training_suite.m_regularization_rate / (float)training_suite.m_training_data.size());
    } else if (training_suite.m_regularization == Regularization::L1) {
        regularizationTerm2Base = -((training_suite.m_learning_rate * (training_suite.m_regularization_rate / (float)training_suite.m_training_data.size())));
    }
    const bool applyRegularizationTerm2 = regularizationTerm2Base != 0.0f;

    const float normalized_learning_rate = training_suite.m_learning_rate * (float(trainingDataEnd - trainingDataBegin) / (float)training_suite.m_training_data.size());

    // apply_gradient
    std::for_each(std::execution::par_unseq, network_layout.begin(), network_layout.end(), [&](const LayerConfig& layer_config) {
        const uint32_t layer_id = &layer_config - &network_layout[0];
        const auto layer_weight_data_offset = GetOffsetToLayerWeights(network, layer_id);
        const auto layer_weight_data = network.GetRawWeightData().data() + layer_weight_data_offset;
        auto layer_gradient_data = accumulated_gradient.data() + layer_weight_data_offset;
        const auto weights_per_neuron = GetLayerWeightsPerNeuronCount(network, layer_id);
        const auto neuron_data_size = weights_per_neuron + 1;

        for (size_t i = 0; i < layer_config.m_num_neurons; ++i) {
            auto neuron_weight_bias_data = layer_weight_data + i * neuron_data_size;
            for (size_t j = 0; j < weights_per_neuron; ++j) {
                auto weight_data = neuron_weight_bias_data + j;
                const auto g = *(layer_gradient_data++);
                *weight_data = regularizationTerm1 * (*weight_data) - g * normalized_learning_rate;
                if (applyRegularizationTerm2) {
                    *weight_data -= regularizationTerm2Base * sign(*weight_data);
                }
            }
            *(neuron_weight_bias_data + weights_per_neuron) -= *(layer_gradient_data++) * normalized_learning_rate; // bias
        }

        ASSERT(layer_gradient_data - (accumulated_gradient.data() + layer_weight_data_offset) == (GetOffsetToLayerWeights(network, layer_id + 1) - GetOffsetToLayerWeights(network, layer_id)));
    });
}

std::vector<float> CPUComputeDevice::CalculateAccumulatedGradientForBatch(const NetworkResourceHandle& network_handle, const TrainingSuite& training_suite, uint32_t batch_begin,
                                                                          uint32_t batch_end) const
{
    const Network& network = *network_handle.m_network;
    const uint32_t total_neurons = network.GetNeuronCount();

    std::vector<float> accumulated_gradient;
    accumulated_gradient.resize(network.GetRawWeightData().size());

    std::vector<float> delta_k_buffer;
    delta_k_buffer.resize(CalculateLargestLayerNeuronCount(network.GetLayerConfig()));

    std::optional<InterimTrainingData> interim_data = InterimTrainingData{total_neurons};

    std::vector<float> output_buffer; // buffer for output values
    output_buffer.resize(network.GetOutputCount(), 0.0f);

    for (int i = batch_begin; i < batch_end; ++i) {
        const auto& training_input = training_suite.m_training_data[i].m_input;
        const auto& desired_output = training_suite.m_training_data[i].m_desired_output;
        EvaluateAndCollectInterimData(output_buffer, network_handle, training_input, interim_data);

        CalculateOutputLayerGradient(network, training_suite.m_cost_function, accumulated_gradient, delta_k_buffer, *interim_data, training_input, desired_output);

        for (int lyr_id = network.GetLayerConfig().size() - 2; lyr_id >= 0; --lyr_id) {
            CalculateHiddenLayerGradient(network, lyr_id, accumulated_gradient, delta_k_buffer, *interim_data, training_input);
        }
    }

    return accumulated_gradient;
}

void CPUComputeDevice::CalculateOutputLayerGradient(const Network& network, CostFunction cost_function, std::span<float> gradient_data, std::span<float> delta_k_vector,
                                                    const InterimTrainingData& interim_data, const std::vector<float>& training_input, const std::vector<float>& desired_output) const
{
    // Read only data
    const int last_layer_idx = network.GetLayerConfig().size() - 1;
    const auto activationFunction = network.GetLayerConfig()[last_layer_idx].m_activation;
    const uint32_t last_layer_neuron_count = network.GetLayerConfig()[last_layer_idx].m_num_neurons;
    const uint32_t last_layer_weight_count = last_layer_idx == 0 ? network.GetInputCount() : network.GetLayerConfig()[last_layer_idx - 1].m_num_neurons;
    std::span<const float> prev_activations; // Activations in the previous layer
    if (last_layer_idx == 0) {
        prev_activations = training_input;
    } else {
        const uint32_t prev_layer_neuron_count = last_layer_weight_count;
        const size_t begin = interim_data.m_activations.size() - last_layer_neuron_count - prev_layer_neuron_count;
        const size_t end = interim_data.m_activations.size() - last_layer_neuron_count;
        prev_activations = std::span<const float>(&interim_data.m_activations[begin], &interim_data.m_activations[end]);
    }
    std::span<const float> last_layer_activations =
        std::span<const float>(interim_data.m_activations.end() - last_layer_neuron_count, interim_data.m_activations.end());                                    // Activations in the last layer
    std::span<const float> last_layer_z_values = std::span<const float>(interim_data.m_z_values.end() - last_layer_neuron_count, interim_data.m_z_values.end()); // Z Values in the last layer
    const size_t last_layer_weight_and_bias_count = last_layer_neuron_count * (last_layer_weight_count + 1);

    // Write data
    std::span<float> gradient_last_layer = std::span<float>(gradient_data.end() - last_layer_weight_and_bias_count, gradient_data.end());

    std::for_each_n(std::execution::par_unseq, gradient_last_layer.begin(), last_layer_neuron_count, [&](const float& it) {
        const uint32_t i = &it - &gradient_last_layer[0];

        const float outputValue = last_layer_activations[i];
        const float delta_k = CalculateCostFunctionDelta(cost_function, last_layer_z_values[i], outputValue, desired_output[i], activationFunction);
        float* gradient_neuron_data = gradient_last_layer.data() + i * (last_layer_weight_count + 1);

        for (int j = 0; j < last_layer_weight_count; j++) {
            gradient_neuron_data[j] += delta_k * prev_activations[j];
        }
        gradient_neuron_data[last_layer_weight_count] += delta_k;
        delta_k_vector[i] = delta_k;
    });
}

void CPUComputeDevice::CalculateHiddenLayerGradient(const Network& network, uint32_t layer_id, std::span<float> gradient_data, std::span<float> delta_k_vector, const InterimTrainingData& interim_data,
                                                    const std::vector<float>& training_input) const
{
    // Read only data
    const ActivationFunction activation_fnc = network.GetLayerConfig()[layer_id].m_activation;
    const uint32_t layer_weight_count = GetLayerWeightsPerNeuronCount(network, layer_id);
    const uint32_t layer_neuron_count = network.GetLayerConfig()[layer_id].m_num_neurons;
    const uint32_t next_layer_neuron_count = network.GetLayerConfig()[layer_id + 1].m_num_neurons;
    std::span<const float> next_layer_weights = std::span<const float>(network.GetRawWeightData().begin() + GetOffsetToLayerWeights(network, layer_id + 1),
                                                                       network.GetRawWeightData().begin() + GetOffsetToLayerWeights(network, layer_id + 2));

    std::span<const float> layer_z_values = std::span<const float>(interim_data.m_z_values.begin() + GetOffsetToLayerNeuronCount(network.GetLayerConfig(), layer_id),
                                                                   interim_data.m_z_values.begin() + GetOffsetToLayerNeuronCount(network.GetLayerConfig(), layer_id + 1));
    std::span<const float> prev_layer_activations = layer_id == 0 ? training_input
                                                                  : std::span<const float>(interim_data.m_activations.begin() + GetOffsetToLayerNeuronCount(network.GetLayerConfig(), layer_id - 1),
                                                                                           interim_data.m_activations.begin() + GetOffsetToLayerNeuronCount(network.GetLayerConfig(), layer_id));

    // Write data:
    std::span<float> current_layer_gradient_data =
        std::span<float>(gradient_data.begin() + GetOffsetToLayerWeights(network, layer_id), gradient_data.begin() + GetOffsetToLayerWeights(network, layer_id + 1));

    std::vector<float> newGammak;
    newGammak.resize(layer_neuron_count);

    std::for_each_n(std::execution::par_unseq, current_layer_gradient_data.begin(), layer_neuron_count, [&](const float& it) {
        const uint32_t i = &it - &current_layer_gradient_data[0];

        float* gradient_neuron_data = current_layer_gradient_data.data() + i * (layer_weight_count + 1);

        float deltak = 0;
        const uint32_t next_layer_neuron_data_size = layer_neuron_count + 1; // weights + bias
        const float* next_layer_weight_for_ith_neuron = next_layer_weights.data() + i;
        for (uint32_t k = 0; k < next_layer_neuron_count; ++k) {
            deltak += delta_k_vector[k] * next_layer_weight_for_ith_neuron[k * next_layer_neuron_data_size];
        }
        deltak *= CalculateActivationFunctionPrime(activation_fnc, layer_z_values[i]);
        newGammak[i] = deltak;

        for (int j = 0; j < layer_weight_count; ++j) {
            gradient_neuron_data[j] += deltak * (prev_layer_activations[j]);
        }
        gradient_neuron_data[layer_weight_count] += deltak; // bias
    });

    memcpy(delta_k_vector.data(), newGammak.data(), newGammak.size() * sizeof(float));
}

void CPUComputeDevice::EvaluateAndCollectInterimData(std::span<float> result_buffer, const NetworkResourceHandle& network_handle, std::span<const float> input,
                                                     std::optional<InterimTrainingData>& output_interim_data) const
{
    Network& network = *network_handle.m_network;

    if (input.size() != network.GetInputCount()) {
        throw std::runtime_error("Invalid input length!");
    }

    if (result_buffer.size() != network.GetOutputCount()) {
        throw std::runtime_error("Invalid result buffer length!");
    }

    auto layer_config = network.GetLayerConfig();
    std::vector<float> layer_args = std::vector<float>(input.begin(), input.end());
    std::vector<float> hidden_layer_result_buffer{};
    const float* layer_weight_data = network.GetRawWeightData().data();

    uint64_t activation_offset = 0;

    for (size_t i = 0; i < layer_config.size(); ++i) {
        const bool is_output_layer = i == layer_config.size() - 1;

        const uint32_t input_num = uint32_t(layer_args.size());
        const uint32_t output_num = layer_config[i].m_num_neurons;
        ActivationFunction activation_fnc = layer_config[i].m_activation; // If Z values are required,

        std::span<float> layer_result;

        if (!is_output_layer) {
            hidden_layer_result_buffer.clear();
            hidden_layer_result_buffer.resize(output_num);
            layer_result = hidden_layer_result_buffer;
        } else {
            // Write the result of the last layer directly into the buffer given from the outside
            layer_result = result_buffer;
        }

        std::for_each_n(std::execution::par_unseq, network.GetRawWeightData().begin(), output_num, [&](const float& f) {
            const uint32_t neuron_id = &f - &network.GetRawWeightData()[0];
            float acc = 0.0f;
            const float* neuron_weight_data = layer_weight_data + (input_num + 1) * neuron_id; // pointer to the weights of this neuron
            for (uint32_t weight_id = 0; weight_id < input_num; weight_id++) {
                acc += neuron_weight_data[weight_id] * layer_args[weight_id];
            }
            acc += neuron_weight_data[input_num]; // bias
            // TODO: acc may become too large here in case of large networks, handle NaN!
            layer_result[neuron_id] = output_interim_data ? acc : CalculateActivationFunction(activation_fnc, acc);
        });

        if (output_interim_data) {
            memcpy(output_interim_data->m_z_values.data() + activation_offset, layer_result.data(), layer_result.size() * sizeof(float));

            // If z value output was required, we had to skip the activation function to be able to provide it (by using passtrough). Here we apply the real activation function
            std::for_each(std::execution::par_unseq, layer_result.begin(), layer_result.end(), [activation_fnc](float& f) { f = CalculateActivationFunction(activation_fnc, f); });

            memcpy(output_interim_data->m_activations.data() + activation_offset, layer_result.data(), layer_result.size() * sizeof(float));

            activation_offset += layer_result.size();
        }

        layer_weight_data += input_num * output_num + output_num; // Advance the pointer to the weights of the layer

        if (!is_output_layer) {
            std::swap(layer_args, hidden_layer_result_buffer);
        }
    }

    ASSERTM(layer_weight_data - network.GetRawWeightData().data() == network.GetRawWeightData().size(), "");
}

std::unique_ptr<NetworkResourceHandle> CPUComputeDevice::RegisterNetwork(Network& network) { return std::make_unique<NetworkResourceHandle>(network); }

std::string CPUComputeDevice::GetDeviceName() const
{
    hwinfo::CPU cpu;
    return "CPU device: " + cpu.getModelName();
}

size_t CPUComputeDevice::GetTotalMemory() const
{
    hwinfo::RAM ram;
    return size_t(ram.getTotalSize_Bytes());
}

uint32_t CPUComputeDevice::GetComputeUnits() const
{
    hwinfo::CPU cpu;
    return cpu.getNumLogicalCores();
}

bool CPUComputeDevice::SupportsWeightFormat(NetworkWeightFormat format) const
{
    switch (format) {
    case macademy::NetworkWeightFormat::Float16:
        return true;
    case macademy::NetworkWeightFormat::Float32:
        return true;
    }

    throw std::runtime_error("CPUComputeDevice::SupportsWeightFormat: Invalid NetworkWeightFormat!");
}

std::vector<float> CPUComputeDevice::Evaluate(const NetworkResourceHandle& network_handle, std::span<const float> input) const { return EvaluateBatch(1, network_handle, input); }

std::vector<float> CPUComputeDevice::EvaluateBatch(uint32_t batch_size, const NetworkResourceHandle& network_handle, std::span<const float> input) const
{
    std::optional<InterimTrainingData> empty{};
    const auto output_layer_size = network_handle.m_network->GetOutputCount();
    const auto input_layer_size = network_handle.m_network->GetInputCount();
    std::vector<float> ret;
    ret.resize(output_layer_size * batch_size);

    for (uint32_t i = 0; i < batch_size; ++i) {
        const auto output_offset = output_layer_size * i;
        const auto input_offset = input_layer_size * i;
        EvaluateAndCollectInterimData(std::span<float>(ret.begin() + output_offset, ret.begin() + output_offset + output_layer_size), network_handle,
                                      std::span<const float>(input.begin() + input_offset, input.begin() + input_offset + input_layer_size), empty);
    }

    return ret;
}

void CPUComputeDevice::ApplyRandomMutation(NetworkResourceHandle& network_handle, MutationDistribution weight_mutation_distribution, MutationDistribution bias_mutation_distribution)
{
    Network& network = *network_handle.m_network;

    auto layer_config = network.GetLayerConfig();

    uint32_t weights_per_neuron = network.GetInputCount();
    uint32_t layer_neuron_count = 0;

    std::random_device rd;
    std::mt19937 gen(rd());

    auto generate_mutator = [&](const MutationDistribution& mutation_distribution) {
        if (std::holds_alternative<UniformDistribution>(mutation_distribution)) {
            UniformDistribution uniform_distribution_desc = std::get<UniformDistribution>(mutation_distribution);
            std::uniform_real_distribution uniform_distribution(-uniform_distribution_desc.range, uniform_distribution_desc.range);

            return [uniform_distribution, &gen](float x) mutable { return x + uniform_distribution(gen); };
        }
        throw std::runtime_error("Invalid mutation distribution!");
    };

    std::function<float(float)> weight_mutator = generate_mutator(weight_mutation_distribution);
    std::function<float(float)> bias_mutator = generate_mutator(bias_mutation_distribution);

    float* data_ptr = network.GetRawWeightData().data();

    for (size_t i = 0; i < layer_config.size(); ++i) {
        layer_neuron_count = layer_config[0].m_num_neurons;

        for (uint32_t n = 0; n < layer_neuron_count; ++n) {
            for (uint32_t w = 0; w < weights_per_neuron; ++w) {
                *data_ptr = weight_mutator(*data_ptr);
                ++data_ptr;
            }
            *data_ptr = bias_mutator(*data_ptr);
            ++data_ptr;
        }

        weights_per_neuron = layer_neuron_count;
    }
}
#endif

std::unique_ptr<IBuffer> CPUComputeDevice::CreateBuffer(size_t size, BufferUsage, const std::string& name)
{
    auto ret = std::make_unique<CPUBuffer>();
    ret->m_data.resize(size);

    return ret;
}

void CPUComputeDevice::QueueWriteToBuffer(IBuffer* dst_buffer, std::span<const uint8_t> src, size_t buffer_offset)
{
    CPUBuffer* cpu_buffer = BufferCast<CPUBuffer>(dst_buffer);

    ASSERT(cpu_buffer->m_data.size() >= buffer_offset + src.size());

    memcpy(cpu_buffer->m_data.data() + buffer_offset, src.data(), src.size());
}

void CPUComputeDevice::QueueReadFromBuffer(IBuffer* src_buffer, std::span<uint8_t> dst, size_t buffer_offset)
{
    CPUBuffer* cpu_buffer = BufferCast<CPUBuffer>(src_buffer);

    ASSERT(cpu_buffer->m_data.size() >= buffer_offset + dst.size());

    memcpy(dst.data(), cpu_buffer->m_data.data() + buffer_offset, dst.size());
}

void CPUComputeDevice::QueueFillBuffer(IBuffer* buffer, uint32_t data, size_t offset_bytes, size_t size_bytes)
{
    CPUBuffer* cpu_buffer = BufferCast<CPUBuffer>(buffer);

    ASSERT(cpu_buffer->m_data.size() >= offset_bytes + size_bytes);

    memset(cpu_buffer->m_data.data() + offset_bytes, data, size_bytes);
}

void CPUComputeDevice::SubmitQueue()
{ /*CPU doesn't queue operations*/
}

void CPUComputeDevice::WaitQueueIdle()
{ /*CPU doesn't queue operations*/
}

void CPUComputeDevice::QueueEvaluateLayerBatched(const IBuffer* weights_buffer, const IBuffer* layer_config_buffer, const IBuffer* layer_input_buffer, IBuffer* layer_output_buffer, uint32_t layer_id,
                                                 uint64_t weights_layer_offset, uint32_t batch_count, uint32_t layer_neuron_count)
{
    const auto weights_f32 = BufferCast<const CPUBuffer>(weights_buffer)->As<const float>();
    const auto layer_input = BufferCast<const CPUBuffer>(layer_input_buffer)->As<const float>();
    auto layer_output = BufferCast<CPUBuffer>(layer_output_buffer)->As<float>();
    const auto layer_config = BufferCast<const CPUBuffer>(layer_config_buffer)->As<const uint32_t>();

    ASSERT(layer_config[2 + layer_id * 2] == layer_neuron_count); // value in the buffer should match

    const uint32_t weights_per_neuron = layer_config[layer_id * 2]; // neurons in the prev layer
    const uint32_t activationFunctionId = layer_config[3 + layer_id * 2];

    std::for_each_n(std::execution::par_unseq, weights_f32, layer_neuron_count * batch_count, [&](const float& f) {
        const uint32_t neuron_id = &f - weights_f32;

        const uint32_t layer_neuron_id = neuron_id % layer_neuron_count;
        const uint32_t batch_id = neuron_id / layer_neuron_count;

        const float* input = layer_input + batch_id * weights_per_neuron;
        float* output = layer_output + batch_id * layer_neuron_count;

        const uint32_t neuron_data_size = weights_per_neuron + 1; // weights in prev layer + 1 bias

        const float* neuron_weights_biases = weights_f32 + weights_layer_offset + layer_neuron_id * neuron_data_size;

        float acc = 0;
        for (int i = 0; i < weights_per_neuron; ++i) {
            acc += neuron_weights_biases[i] * input[i];
        }
        acc += neuron_weights_biases[weights_per_neuron]; // bias

        output[layer_neuron_id] = CalculateActivationFunction(ActivationFunction(activationFunctionId), acc);
    });
}

std::string CPUComputeDevice::GetDeviceName() const
{
    hwinfo::CPU cpu;
    return "CPU device: " + cpu.getModelName();
}

size_t CPUComputeDevice::GetTotalMemory() const
{
    hwinfo::RAM ram;
    return size_t(ram.getTotalSize_Bytes());
}

bool CPUComputeDevice::SupportsWeightFormat(NetworkWeightFormat format) const
{
    switch (format) {
    case macademy::NetworkWeightFormat::Float16:
        return true;
    case macademy::NetworkWeightFormat::Float32:
        return true;
    }

    throw std::runtime_error("CPUComputeDevice::SupportsWeightFormat: Invalid NetworkWeightFormat!");
}
void CPUComputeDevice::QueueTrainForwardPass(const IBuffer* weights_buffer, const IBuffer* layer_config_buffer, IBuffer* m_activations_zvalues_buffer, const IBuffer* input_buffer,
                                             uint32_t layer_neuron_count, uint32_t layer_id, uint64_t weights_layer_offset, uint32_t num_training_samples, uint32_t total_neuron_count)
{
    const auto weights_f32 = BufferCast<const CPUBuffer>(weights_buffer)->As<const float>();
    const auto layer_config = BufferCast<const CPUBuffer>(layer_config_buffer)->As<const uint32_t>();
    auto activations_zvalues = BufferCast<CPUBuffer>(m_activations_zvalues_buffer)->As<float>();
    const auto input_data = BufferCast<const CPUBuffer>(input_buffer)->As<const float>();

    const uint32_t weights_per_neuron = layer_config[layer_id * 2]; // neurons in the prev layer
    const uint32_t activationFunctionId = layer_config[3 + layer_id * 2];

    // TODOZ parallel for
    for (size_t g_id = 0; g_id < layer_neuron_count * num_training_samples; ++g_id) {
        const uint32_t layer_neuron_id = g_id % layer_neuron_count;
        const uint32_t trainingSampleId = g_id / layer_neuron_count;

        const uint32_t neuron_data_size = weights_per_neuron + 1; // weights in prev layer + 1 bias

        const float* neuron_weights_biases = weights_f32 + weights_layer_offset + layer_neuron_id * neuron_data_size;

        const int training_sample_activation_offset = total_neuron_count * trainingSampleId;

        const uint32_t input_layer_neuron_count = layer_config[0];
        const float* prevActivations = layer_id == 0 ? (input_data + input_layer_neuron_count * trainingSampleId)
                                                     : (activations_zvalues + (training_sample_activation_offset + GetLayerNeuronCountOffset(layer_id - 1, layer_config)));

        // Calculate ZValues for layer
        float acc = 0;
        for (int i = 0; i < weights_per_neuron; ++i) {
            acc += neuron_weights_biases[i] * prevActivations[i];
        }
        acc += neuron_weights_biases[weights_per_neuron]; // bias

        // Store ZValues and the result of the activation function
        const int layer_activation_offset = training_sample_activation_offset + GetLayerNeuronCountOffset(layer_id, layer_config);
        const int layer_zvalue_offset = layer_activation_offset + (num_training_samples * total_neuron_count); // zvalues are stored after activations, so shift by the number of total activations
        activations_zvalues[layer_zvalue_offset + layer_neuron_id] = acc;
        activations_zvalues[layer_activation_offset + layer_neuron_id] = CalculateActivationFunction(ActivationFunction(activationFunctionId), acc);
    }
}

void CPUComputeDevice::QueueTrainBackwardPass(const IBuffer* weights_buffer, const IBuffer* layer_config_buffer, const IBuffer* m_activations_zvalues_buffer, const IBuffer* input_buffer,
                                              IBuffer* delta_k_vector_buffer, IBuffer* gradient_buffer, const IBuffer* desiredOutputsBuffer, uint32_t layer_neuron_count, uint32_t layer_id,
                                              uint32_t layer_count, uint32_t numTrainingSamples, uint32_t total_neuron_count, CostFunction costFunction, uint32_t largest_layer_neuron_count,
                                              uint64_t layer_weights_offset)
{
    const auto weights_f32 = BufferCast<const CPUBuffer>(weights_buffer)->As<const float>();
    const auto layer_config = BufferCast<const CPUBuffer>(layer_config_buffer)->As<const uint32_t>();
    const auto activations_zvalues = BufferCast<const CPUBuffer>(m_activations_zvalues_buffer)->As<const float>();
    const auto input_data = BufferCast<const CPUBuffer>(input_buffer)->As<const float>();
    auto delta_k_vector = BufferCast<CPUBuffer>(delta_k_vector_buffer)->As<float>();
    auto gradient = BufferCast<CPUBuffer>(gradient_buffer)->As<float>();
    const auto desiredOutputs = BufferCast<const CPUBuffer>(desiredOutputsBuffer)->As<const float>();

    const uint32_t weights_per_neuron = layer_config[layer_id * 2]; // neurons in the prev layer
    const uint32_t activationFunctionId = layer_config[3 + layer_id * 2];
    const uint32_t deltaKVectorStride = largest_layer_neuron_count; // Table size of delta_k vector

    //__constant const int* layerNeuronCountBegin = config+8;

    for (size_t g_id = 0; g_id < layer_neuron_count * numTrainingSamples; ++g_id) {

        const uint32_t layer_neuron_id = g_id % layer_neuron_count;
        const uint32_t trainingSampleId = g_id / layer_neuron_count;

        const int delta_k_read_offset = deltaKVectorStride * 2 * trainingSampleId + ((layer_id % 2) * deltaKVectorStride);
        const int delta_k_write_offset = deltaKVectorStride * 2 * trainingSampleId + (((layer_id + 1) % 2) * deltaKVectorStride);

        const int training_sample_activation_offset = total_neuron_count * trainingSampleId;
        const int current_layer_activation_offset = training_sample_activation_offset + GetLayerNeuronCountOffset(layer_id, layer_config);
        const int current_layer_z_values_offset = current_layer_activation_offset + (numTrainingSamples * total_neuron_count); // shift by table size

        const uint32_t input_layer_neuron_count = layer_config[0];
        const float* prevActivations = layer_id == 0 ? (input_data + input_layer_neuron_count * trainingSampleId)
                                                     : (activations_zvalues + (training_sample_activation_offset + GetLayerNeuronCountOffset(layer_id - 1, layer_config)));

        const float zValue = activations_zvalues[current_layer_z_values_offset + layer_neuron_id];

        float delta_k;

        if (layer_id == (layer_count - 1)) {
            // Output layer
            const float activation = activations_zvalues[current_layer_activation_offset + layer_neuron_id];
            const float desiredOutput = desiredOutputs[trainingSampleId * layer_neuron_count + layer_neuron_id];
            delta_k = CalculateCostFunctionDelta(costFunction, ActivationFunction(activationFunctionId), zValue, activation, desiredOutput);
        } else {
            // Hidden layer
            delta_k = 0;
            const uint32_t next_layer_weights_offset = layer_weights_offset + (layer_neuron_count * (weights_per_neuron + 1));
            const uint32_t next_layer_weight_offset_for_processed_neuron = next_layer_weights_offset + layer_neuron_id;
            const uint32_t next_layer_neuron_count = layer_config[2 + (layer_id + 1) * 2]; // number of neurons
            const uint32_t next_layer_neuron_data_size = layer_neuron_count + 1;           // weights + bias
            for (uint32_t i = 0; i < next_layer_neuron_count; ++i) {
                delta_k += delta_k_vector[delta_k_read_offset + i] * weights_f32[next_layer_weight_offset_for_processed_neuron + (i * next_layer_neuron_data_size)];
            }
            delta_k *= CalculateActivationFunctionPrime(ActivationFunction(activationFunctionId), zValue);
        }

        const uint32_t gradientBaseOffset = layer_weights_offset + layer_neuron_id * (weights_per_neuron + 1);

        for (int i = 0; i < weights_per_neuron; ++i) {
            float& v = *(gradient + gradientBaseOffset + i);
            v += delta_k * prevActivations[i];
        }
        float& v = *(gradient + gradientBaseOffset + weights_per_neuron);
        v += delta_k; // bias

        if (layer_id != 0) {
            delta_k_vector[delta_k_write_offset + layer_neuron_id] = delta_k;
        }
    }
}

void CPUComputeDevice::QueueApplyGradients(IBuffer* weights_buffer, const IBuffer* gradient_buffer, const IBuffer* layer_config_buffer, uint32_t layer_neuron_count, uint32_t layer_id,
                                           uint64_t layer_weight_data_offset, float regularization_term_1, float regularization_term_2, float normalized_learning_rate)
{
    auto weights_f32 = BufferCast<CPUBuffer>(weights_buffer)->As<float>();
    const auto gradient = BufferCast<const CPUBuffer>(gradient_buffer)->As<const float>();
    const auto layer_config = BufferCast<const CPUBuffer>(layer_config_buffer)->As<const uint32_t>();

    const uint32_t weights_per_neuron = layer_config[layer_id * 2]; // neurons in the prev layer
    const bool applyRegularizationTerm2 = regularization_term_2 != 0.0f;

    const auto layer_weight_data = weights_f32 + layer_weight_data_offset;
    auto layer_gradient_data = gradient + layer_weight_data_offset;
    const auto neuron_data_size = weights_per_neuron + 1;

    // TODO parallel for
    for (size_t i = 0; i < layer_neuron_count; ++i) {
        auto neuron_weight_bias_data = layer_weight_data + i * neuron_data_size;
        for (size_t j = 0; j < weights_per_neuron; ++j) {
            auto weight_data = neuron_weight_bias_data + j;
            const auto g = *(layer_gradient_data++);
            *weight_data = regularization_term_1 * (*weight_data) - g * normalized_learning_rate;
            if (applyRegularizationTerm2) {
                *weight_data -= regularization_term_2 * sign(*weight_data);
            }
        }
        *(neuron_weight_bias_data + weights_per_neuron) -= *(layer_gradient_data++) * normalized_learning_rate; // bias
    }
}

ComputeDeviceInfo CPUComputeDevice::GetCpuComputeDeviceInfo()
{
    hwinfo::CPU cpu;
    hwinfo::RAM ram;
    return ComputeDeviceInfo{.m_backend = "cpu", .m_device_index = 0, .m_device_name = cpu.getModelName(), .m_total_memory = uint64_t(ram.getTotalSize_Bytes())};
}

} // namespace macademy