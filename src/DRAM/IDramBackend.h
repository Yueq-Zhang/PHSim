#pragma once

#include <cstdint>

struct MemoryAccess;

// Common request/response data path shared by the cycle-accurate and
// event-driven DRAM backends. Timing progression intentionally remains in the
// concrete backends because the two models advance time differently.
class IDramBackend {
   public:
    virtual ~IDramBackend() = default;

    virtual bool running() = 0;
    virtual bool is_full(uint32_t cid, MemoryAccess* request) = 0;
    virtual void push(uint32_t cid, MemoryAccess* request) = 0;
    virtual bool is_empty(uint32_t cid) = 0;
    virtual MemoryAccess* top(uint32_t cid) = 0;
    virtual void pop(uint32_t cid) = 0;
    virtual uint32_t get_channel_id(MemoryAccess* request) = 0;
};
