#pragma once

#include "opencl_common.h"
#include "common.h"
#include <memory>

namespace macademy {
class OpenCLBuffer
{
    std::unique_ptr<cl::Buffer> m_buffer;
    const size_t m_size = 0;

  public:
    OpenCLBuffer(cl::Context& context, cl_mem_flags flags, size_t size, void* host_ptr = nullptr) : m_size(size)
    {
        cl_int err;
        m_buffer = std::make_unique<cl::Buffer>(context, flags, size, host_ptr, &err);

        if (err != 0) {
            m_buffer.reset();
            throw std::runtime_error("Failed to create buffer!");
        }
    }

    void UploadData(cl::CommandQueue& queue, size_t offset, std::span<uint8_t> data, bool blocking)
    {
        if (m_size == 0) {
            throw std::runtime_error("Failed to upload to empty buffer!");
        }

        queue.enqueueWriteBuffer(*m_buffer, blocking, offset, data.size_bytes(), data.data());
    }

    size_t GetSize() const { return m_size; }

    cl::Buffer& GetBuffer() { return *m_buffer; }
};
} // namespace macademy