#pragma once

#include <iostream>
#include <chrono>
#include <functional>
#include <optional>

#include "network.h"
#include "default_weight_initializer.h"
#include "training_suite.h"
#include "training.h"
#include "i_compute_device.h"
#include "compute_tasks.h"
#include <set>
#include <chrono>
#include <map>
#include <cmath>

using namespace std::chrono_literals;
using namespace macademy;

namespace macademy {

struct NetworkResourceHandle;

class ConsoleApp
{
  public:
    struct Command
    {
        std::string m_description;
        std::function<bool(const std::vector<std::string>&)> m_handler;
    };

  protected:
    std::vector<ComputeDeviceInfo> m_devices;
    std::unordered_map<std::string, Command> m_commands;
    ComputeDeviceInfo m_selected_device_info;
    std::unique_ptr<IComputeDevice> m_compute_device;
    std::unique_ptr<NetworkResourceHandle> m_network_resources;
    ComputeTasks m_compute_tasks;
    std::unique_ptr<Network> m_network;

    std::vector<std::string> Split(const std::string& src, const char delimiter);

    void TrainingDisplay(const TrainingResultTracker& tracker);

    void EnsureNetworkResources();

  public:
    void AddCommand(const std::string& name, Command&& cmd) { m_commands[name] = std::move(cmd); }

    ConsoleApp();

    void Run();
};

} // namespace macademy