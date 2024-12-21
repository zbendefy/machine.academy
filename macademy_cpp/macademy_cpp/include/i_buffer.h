#pragma once
#include <span>
#include <memory>

namespace macademy {

class IBuffer
{
  public:
    virtual ~IBuffer() {}

    virtual size_t GetSize() const = 0;
};

template <typename T> T* BufferCast(IBuffer* i_buf)
{
    T* ret = dynamic_cast<T*>(i_buf);
    if (!ret) {
        throw std::exception("Invalid buffer cast!");
    }

    return ret;
}

template <typename T> const T* BufferCast(const IBuffer* i_buf)
{
    T* ret = dynamic_cast<const T*>(i_buf);
    if (!ret) {
        throw std::exception("Invalid buffer cast!");
    }

    return ret;
}

enum class BufferUsage
{
    ReadOnly,
    ReadWrite,
    WriteOnly
};

} // namespace macademy