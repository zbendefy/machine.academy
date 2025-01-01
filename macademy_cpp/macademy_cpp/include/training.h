#pragma once

#include "network.h"
#include "training_suite.h"

#include <algorithm>

namespace macademy {

struct NetworkResourceHandle;

class Training
{
  public:
    std::shared_ptr<const TrainingResultTracker> Train(NetworkResourceHandle& network, std::shared_ptr<TrainingSuite> training_suite);
};

} // namespace macademy