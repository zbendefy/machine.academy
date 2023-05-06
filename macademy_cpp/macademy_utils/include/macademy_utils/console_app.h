#pragma once

#include <iostream>
#include <chrono>
#include <functional>

#include "network.h"
#include "default_weight_initializer.h"
#include "cpu_compute_backend.h"
#include "opencl_backend/opencl_compute_device.h"
#include "training_suite.h"
#include "training.h"
#include <set>
#include <chrono>
#include <map>
#include <cmath>

using namespace std::chrono_literals;
using namespace macademy;

namespace macademy {

class ConsoleApp
{
  public:
    struct Command
    {
        std::string m_description;
        std::function<bool(const std::vector<std::string>&)> m_handler;
    };

  protected:
    std::vector<std::unique_ptr<IComputeDevice>> m_devices;
    std::map<IComputeDevice*, std::unique_ptr<NetworkResourceHandle>> m_uploaded_networks;
    std::unordered_map<std::string, Command> m_commands;
    IComputeDevice* m_selected_device = nullptr;

    std::vector<std::string> Split(const std::string& src, const char delimiter);

    void TrainingDisplay(const TrainingResultTracker& tracker);

  public:
    void AddCommand(const std::string& name, Command&& cmd) { m_commands[name] = std::move(cmd); }

    ConsoleApp();

    void Run();
};

} // namespace macademy