#include "compute_tasks.h"
#include "i_compute_device.h"
#include "network.h"
#include "common.h"
#include "utils.h"
#include "training_suite.h"

#include <fstream>
#include <sstream>

namespace {
size_t ExtendGlobalWorkSize(size_t desiredGlobalSize, size_t localSize)
{
    return ((desiredGlobalSize % localSize) == 0) ? desiredGlobalSize : (desiredGlobalSize + (localSize - (desiredGlobalSize % localSize)));
}
} // namespace

namespace macademy {

NetworkResourceHandle::NetworkResourceHandle(Network& network, IComputeDevice& compute_device) : m_network(&network), m_compute_device(&compute_device)
{
    int tensor_id = 0;
    for (const auto& layer : network.GetLayers())
    {
        m_tensor_buffers.emplace_back(m_compute_device->CreateBuffer(layer.m_tensor->GetByteSize(), BufferUsage::ReadWrite, "tensor_" + std::to_string(tensor_id)));
        m_compute_device->QueueWriteToBuffer(m_tensor_buffers.back().get(), ToReadOnlyUi8Span(layer.m_tensor->GetRawData()), 0);
        ++tensor_id;
    }

    m_compute_device->SubmitQueue();
    m_compute_device->WaitQueueIdle();

    // TODO: move this note to opencl createbuffer
    //  Note: using COPY_HOST_PTR at buffer creation is not optimal, as it will copy it first to host memory, and then upload it at kernel runtime
    //  Enqueueing a writebuffer separately makes sure there are no 2 copies, and that the buffer is uploaded when this function returns.
    //  See: https://stackoverflow.com/questions/3832963/what-is-the-difference-between-creating-a-buffer-object-with-clcreatebuffer-cl
}

void NetworkResourceHandle::SynchronizeNetworkData()
{
    for (uint32_t i = 0; i < m_network->GetLayerCount(); ++i)
    {
        m_compute_device->QueueReadFromBuffer(m_tensor_buffers[i].get(), m_network->GetLayers()[i].m_tensor->GetRawData(), 0);
    }
    m_compute_device->SubmitQueue();
    m_compute_device->WaitQueueIdle();
}

void NetworkResourceHandle::AllocateBatchEvalResources() const
{
    const size_t largest_tensor_element_size = sizeof(float);

    const size_t largest_layer_size_bytes = std::max(m_network->GetInputCount(), CalculateLargestLayerNeuronCount(m_network->GetLayers())) * largest_tensor_element_size;
    const size_t largest_layer_buffer_required_size = largest_layer_size_bytes;

    if (!m_layer_result_buffer_a || m_layer_result_buffer_a->GetSize() < largest_layer_buffer_required_size) {
        m_layer_result_buffer_a.reset();
        m_layer_result_buffer_a = m_compute_device->CreateBuffer(largest_layer_buffer_required_size, BufferUsage::ReadWrite, "layer_result_buffer_a");
    }

    if (!m_layer_result_buffer_b || m_layer_result_buffer_b->GetSize() < largest_layer_buffer_required_size) {
        m_layer_result_buffer_b.reset();
        m_layer_result_buffer_b = m_compute_device->CreateBuffer(largest_layer_buffer_required_size, BufferUsage::ReadWrite, "layer_result_buffer_b");
    }
}

void NetworkResourceHandle::AllocateMutationBuffer()
{
    throw "TODOZ";
    if (!m_mutation_buffer) {
        //m_mutation_buffer = m_compute_device->CreateBuffer(m_network->GetRawWeightData().size_bytes(), BufferUsage::ReadOnly, "mutation_buffer");
    }
}

void NetworkResourceHandle::AllocateTrainingResources(uint32_t training_sample_count)
{
    const auto largest_layer_neuron_count = CalculateLargestLayerNeuronCount(m_network->GetLayers());

    m_input_buffer = m_compute_device->CreateBuffer(training_sample_count * m_network->GetInputCount() * sizeof(float), BufferUsage::ReadOnly, "input_buffer");
    m_desired_output_buffer = m_compute_device->CreateBuffer(training_sample_count * m_network->GetOutputCount() * sizeof(float), BufferUsage::ReadOnly, "desired_output_buffer");
    m_delta_k_buffer_a = m_compute_device->CreateBuffer(training_sample_count * largest_layer_neuron_count * sizeof(float), BufferUsage::ReadWrite, "delta_k_buffer_a");
    m_delta_k_buffer_b = m_compute_device->CreateBuffer(training_sample_count * largest_layer_neuron_count * sizeof(float), BufferUsage::ReadWrite, "delta_k_buffer_b");
    for (uint32_t i = 0; i < m_network->GetLayerCount(); ++i)
    {
        m_gradient_buffers.emplace_back(m_compute_device->CreateBuffer(m_network->GetLayers()[i].m_tensor->GetByteSize(), BufferUsage::ReadWrite, "gradient_buffer"));
        m_activation_buffers.emplace_back(m_compute_device->CreateBuffer(training_sample_count * m_network->GetLayers()[i].m_tensor->GetElementSize() * sizeof(float), BufferUsage::ReadWrite, "activations_buffer"));
        m_zvalue_buffers.emplace_back(m_compute_device->CreateBuffer(training_sample_count * m_network->GetLayers()[i].m_tensor->GetElementSize() * sizeof(float), BufferUsage::ReadWrite, "zvalues_buffer"));
    }
}

void NetworkResourceHandle::FreeCachedResources()
{
    m_input_buffer.reset();
    m_desired_output_buffer.reset();
    m_activation_buffers.clear();
    m_zvalue_buffers.clear();
    m_delta_k_buffer_a.reset();
    m_delta_k_buffer_b.reset();
    m_gradient_buffers.clear();
    m_layer_result_buffer_a.reset();
    m_layer_result_buffer_b.reset();
    m_mutation_buffer.reset();
}

std::vector<float> ComputeTasks::Evaluate(const NetworkResourceHandle& network_resources, std::span<const float> input) const
{
    Network& network = *network_resources.m_network;
    IComputeDevice& compute_device = *network_resources.m_compute_device;

    if (input.size() != size_t(network.GetInputCount())) {
        throw std::runtime_error("Invalid input length!");
    }

    network_resources.AllocateBatchEvalResources();

    auto layers = network.GetLayers();

    auto layer_results_input = network_resources.m_layer_result_buffer_a.get();
    auto layer_results_output = network_resources.m_layer_result_buffer_b.get();

    // Write input into buffer for all batches
    compute_device.QueueWriteToBuffer(network_resources.m_layer_result_buffer_a.get(), ToReadOnlyUi8Span(input), 0);

    uint64_t weights_layer_offset = 0;

    for (uint32_t i = 0; i < layers.size(); ++i) {
        const uint32_t input_num = i == 0 ? network.GetInputCount() : layers[i - 1].m_num_neurons;
        const uint32_t output_num = layers[i].m_num_neurons;
        const ActivationFunction activation = layers[i].m_activation;

        compute_device.QueueEvaluateLayer(network_resources.m_tensor_buffers[i].get(), layer_results_input, layer_results_output, activation, input_num, output_num);

        const uint64_t layer_weight_size_bytes = uint64_t(input_num) * output_num + output_num;
        ASSERTM(weights_layer_offset + layer_weight_size_bytes > weights_layer_offset, "Layer weights offset overflow!");
        weights_layer_offset += layer_weight_size_bytes; // advance the offset in the weights buffer for the next layer

        std::swap(layer_results_input, layer_results_output); // output of this layer is input of the next
    }

    std::vector<float> result;
    result.resize(size_t(network.GetOutputCount()));

    auto final_layer_results = layer_results_input;
    compute_device.QueueReadFromBuffer(final_layer_results, ToWriteableUi8Span(result), 0);
    compute_device.SubmitQueue();
    compute_device.WaitQueueIdle();

    return result;
}

void ComputeTasks::TrainMinibatch(NetworkResourceHandle& network_handle, const TrainingSuite& training_suite, uint64_t trainingDataBegin, uint64_t trainingDataEnd) const
{
    Network& network = *network_handle.m_network;
    IComputeDevice& compute_device = *network_handle.m_compute_device;

    const uint32_t num_training_samples = trainingDataEnd - trainingDataBegin;
    auto layers = network.GetLayers();
    const uint32_t total_neuron_count = network.GetNeuronCount();
    const auto largest_layer_neuron_count = CalculateLargestLayerNeuronCount(layers);

    for(auto& gradient_buffer : network_handle.m_gradient_buffers)
    {
        compute_device.QueueFillBuffer(gradient_buffer.get(), 0, 0, gradient_buffer->GetSize());
    }

    std::vector<float> training_input_buffer_data;
    {
        training_input_buffer_data.resize(num_training_samples * network.GetInputCount());
        auto data_ptr = training_input_buffer_data.data();
        for (auto i = trainingDataBegin; i < trainingDataEnd; ++i) {
            std::memcpy(data_ptr, training_suite.m_training_data[i].m_input.data(), training_suite.m_training_data[i].m_input.size() * sizeof(float));
            data_ptr += training_suite.m_training_data[i].m_input.size();
        }

        compute_device.QueueWriteToBuffer(network_handle.m_input_buffer.get(), ToReadOnlyUi8Span(training_input_buffer_data), 0);
    }

    std::vector<float> training_desired_output_buffer_data;
    {
        training_desired_output_buffer_data.resize(num_training_samples * network.GetOutputCount());
        auto data_ptr = training_desired_output_buffer_data.data();
        for (auto i = trainingDataBegin; i < trainingDataEnd; ++i) {
            std::memcpy(data_ptr, training_suite.m_training_data[i].m_desired_output.data(), training_suite.m_training_data[i].m_desired_output.size() * sizeof(float));
            data_ptr += training_suite.m_training_data[i].m_desired_output.size();
        }

        compute_device.QueueWriteToBuffer(network_handle.m_desired_output_buffer.get(), ToReadOnlyUi8Span(training_desired_output_buffer_data), 0);
    }

    // Forward pass (calculating z values and activations for each neuron times for each training data in the network)
    for (uint32_t i = 0; i < layers.size(); ++i) {
        const uint32_t input_num = i == 0 ? network.GetInputCount() : layers[i - 1].m_num_neurons;
        const uint32_t output_num = layers[i].m_num_neurons;
        const bool is_first_layer = i == 0;

        compute_device.QueueTrainForwardPass(network_handle.m_tensor_buffers[i].get(), is_first_layer ? network_handle.m_input_buffer.get() : network_handle.m_activation_buffers[i - 1].get(), is_first_layer, network_handle.m_activation_buffers[i].get(), network_handle.m_zvalue_buffers[i].get(),
            layers[i].m_activation, output_num, input_num, num_training_samples);
    }

    auto delta_k_buffer_read = network_handle.m_delta_k_buffer_a.get();
    auto delta_k_buffer_write = network_handle.m_delta_k_buffer_b.get();

    // Backwards pass (accumulated gradient calculation)
    for (int i = layers.size() - 1; i >= 0; --i) {
        const uint32_t input_num = i == 0 ? network.GetInputCount() : layers[i - 1].m_num_neurons;
        const uint32_t output_num = layers[i].m_num_neurons;
        const bool is_output_layer = i == layers.size() - 1;
        const uint32_t next_layer_neuron_count = is_output_layer ? 0 : layers[i + 1].m_num_neurons;
        const bool is_input_layer = i == 0;

        compute_device.QueueTrainBackwardPass(is_output_layer ? nullptr : network_handle.m_tensor_buffers[i+1].get(), is_input_layer ? network_handle.m_input_buffer.get() : network_handle.m_activation_buffers[i - 1].get(), is_input_layer,
                                              network_handle.m_activation_buffers[i].get(), network_handle.m_zvalue_buffers[i].get(), delta_k_buffer_write, delta_k_buffer_read, network_handle.m_gradient_buffers[i].get(),
                                              network_handle.m_desired_output_buffer.get(), output_num, input_num, layers[i].m_activation, num_training_samples,
                                              training_suite.m_cost_function, next_layer_neuron_count);

        std::swap(delta_k_buffer_write, delta_k_buffer_read);
    }

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

    // Gradient apply pass
    for (uint32_t i = 0; i < layers.size(); ++i) {
        const uint32_t input_num = i == 0 ? network.GetInputCount() : layers[i - 1].m_num_neurons;
        const uint32_t output_num = layers[i].m_num_neurons;

        compute_device.QueueApplyGradients(network_handle.m_tensor_buffers[i].get(), network_handle.m_gradient_buffers[i].get(), output_num, input_num,
            regularizationTerm1, regularizationTerm2Base, normalized_learning_rate);
    }

    compute_device.SubmitQueue();
    compute_device.WaitQueueIdle();
}

void ComputeTasks::ApplyRandomMutation(NetworkResourceHandle& network_handle, MutationDistribution weight_mutation_distribution, MutationDistribution bias_mutation_distribution)
{
    Network& network = *network_handle.m_network;
    IComputeDevice& compute_device = *network_handle.m_compute_device;

#if 0
    network_handle.AllocateMutationBuffer();

    std::vector<float> mutation_buffer_data;
    mutation_buffer_data.resize(network.GetTotalWeightAndBiasCount());

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

    const auto& layer_config = network.GetLayerConfig();

    for (size_t i = 0; i < layer_config.size(); ++i) {
        layer_neuron_count = layer_config[0].m_num_neurons;

        for (uint32_t n = 0; n < layer_neuron_count; ++n) {
            for (uint32_t w = 0; w < weights_per_neuron; ++w) {
                mutation_buffer_data.emplace_back(weight_mutator(0));
            }
            mutation_buffer_data.emplace_back(bias_mutator(0));
        }

        weights_per_neuron = layer_neuron_count;
    }

    compute_device.QueueWriteToBuffer(network_handle.m_mutation_buffer.get(), ToReadOnlyUi8Span(mutation_buffer_data), 0);

    uint64_t weights_layer_offset = 0;

    for (uint32_t i = 0; i < layer_config.size(); ++i) {
        const uint32_t input_num = i == 0 ? network.GetInputCount() : layer_config[i - 1].m_num_neurons;
        const uint32_t output_num = layer_config[i].m_num_neurons;

        compute_device.QueueApplyGradients(network_handle.m_weights.get(), network_handle.m_mutation_buffer.get(), network_handle.m_layer_config_buffer.get(), output_num, i, weights_layer_offset,
                                           1.0f, 0.0f, -1.0f /*note: regularization_term_1 and 2 and learning rate are set to passtrough the modification*/);

        const uint64_t layer_weight_size = uint64_t(input_num) * output_num + output_num;
        ASSERTM(weights_layer_offset + layer_weight_size > weights_layer_offset, "Layer weights offset overflow!");
        weights_layer_offset += layer_weight_size; // advance the offset in the weights buffer for the next layer
    }

    compute_device.SubmitQueue();
    compute_device.WaitQueueIdle();
#endif
}

} // namespace macademy