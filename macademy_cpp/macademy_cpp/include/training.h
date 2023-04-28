#pragma once

#include "network.h"
#include "training_suite.h"

#include <future>
#include <atomic>

namespace macademy
{
    struct TrainigResultTracker
    {
        std::atomic<float> m_epoch_progress = 0;
        std::atomic<uint64_t> m_epochs_finished = 0;
        std::atomic<bool> m_stop_at_next_epoch = false;
        std::future<int64_t> m_future;
    };

    class Training
    {
        TrainigResultTracker Train(const TrainingSuite& suite)
        {
            return {};
        }
    };
}