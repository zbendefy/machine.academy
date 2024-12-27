#pragma once

#include "network.h"
#include "training_suite.h"
#include "compute_tasks.h"

#include <algorithm>

namespace macademy {

class Training
{
  public:
    std::shared_ptr<const TrainingResultTracker> Train(NetworkResourceHandle& network, std::shared_ptr<TrainingSuite> training_suite)
    {
        auto training_result_tracker = std::make_shared<TrainingResultTracker>();

        if (training_suite->m_epochs < 1 || training_suite->m_training_data.empty()) {
            return training_result_tracker;
        }

        if (training_suite->m_training_data[0].m_input.size() != network.m_network->GetInputCount()) {
            throw std::runtime_error("Invalid training input size!");
        }

        if (training_suite->m_training_data[0].m_desired_output.size() != network.m_network->GetOutputCount()) {
            throw std::runtime_error("Invalid training desired output size!");
        }

        training_result_tracker->m_future = std::async(std::launch::async, [this, training_suite, &network, training_result_tracker]() {
            std::random_device rd;
            std::mt19937 g(rd());

            std::vector<TrainingData> training_data_shuffle_buffer; // TODO: threadlocal + clear
            std::span<const TrainingData> training_data_view;

            if (training_suite->m_shuffle_training_data) {
                training_data_shuffle_buffer.resize(training_suite->m_training_data.size());
                std::copy(training_suite->m_training_data.begin(), training_suite->m_training_data.end(), std::back_inserter(training_data_shuffle_buffer));
                training_data_view = training_data_shuffle_buffer;
            } else {
                training_data_view = training_suite->m_training_data;
            }

            network.AllocateTrainingResources(training_suite->m_mini_batch_size ? *training_suite->m_mini_batch_size : training_suite->m_training_data.size());

            for (uint32_t currentEpoch = 0; currentEpoch < training_suite->m_epochs; currentEpoch++) {
                if (training_result_tracker->m_stop_at_next_epoch) {
                    return currentEpoch;
                }

                if (training_suite->m_shuffle_training_data) {
                    std::shuffle(training_data_shuffle_buffer.begin(), training_data_shuffle_buffer.end(), g);
                    training_data_view = training_suite->m_training_data; // update span after shuffling the source vector (todo: this might not be necessary)
                }

                uint64_t trainingDataBegin = 0;
                uint64_t trainingDataEnd =
                    training_suite->m_mini_batch_size ? std::min(*training_suite->m_mini_batch_size, training_suite->m_training_data.size()) : training_suite->m_training_data.size();

                auto compute_device = network.GetComputeDevice();

                while (true) {
                    // compute_device->Train(network, *training_suite, trainingDataBegin, trainingDataEnd);

                    if (training_suite->m_mini_batch_size) {
                        if (trainingDataEnd >= training_suite->m_training_data.size()) {
                            break;
                        }

                        training_result_tracker->m_epoch_progress = float(trainingDataEnd) / training_suite->m_training_data.size();

                        trainingDataBegin = trainingDataEnd;
                        trainingDataEnd = std::min(trainingDataEnd + *training_suite->m_mini_batch_size, training_suite->m_training_data.size());
                    } else {
                        break;
                    }
                }

                ++training_result_tracker->m_epochs_finished;
            }

            network.SynchronizeNetworkData();
            network.FreeCachedResources();

            return training_suite->m_epochs;
        });

        return training_result_tracker;
    }
};

} // namespace macademy