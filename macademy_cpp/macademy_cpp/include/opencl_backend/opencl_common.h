#pragma once

#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 100
#define CL_HPP_CL_1_2_DEFAULT_BUILD 1
#define CL_HPP_ENABLE_EXCEPTIONS 1

#if __has_include(<CL/opencl.hpp>)
#include <CL/opencl.hpp>
#else
// Note: For some reason some opencl installations do not provide opencl.hpp, only cl.hpp
#include <CL/cl.hpp>
#endif