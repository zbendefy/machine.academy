#pragma once

#include "network.h"
#include "training_suite.h"

namespace macademy {
class Training
{
    TrainingResultTracker Train(Network& network, IComputeDevice& compute_device, const TrainingSuite& suite) { return {}; }
};
} // namespace macademy