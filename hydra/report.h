// The renderer the delegate opened, and how it reports errors to the host.
#pragma once

#include "pika.h"

#include <pxr/base/tf/diagnostic.h>

#include <atomic>
#include <cstdio>
#include <string>

PXR_NAMESPACE_OPEN_SCOPE

// husk drops every Tf diagnostic, so errors also go to stderr.
inline void fail(std::string const &message) {
    fprintf(stderr, "pika: error: %s\n", message.c_str());
    TF_RUNTIME_ERROR("Pika: %s", message.c_str());
}

inline void warn(std::string const &message) {
    fprintf(stderr, "pika: warning: %s\n", message.c_str());
    TF_WARN("Pika: %s", message.c_str());
}

// The delegate's renderer, which its prims and passes call for as long as
// the delegate lives.
class Pika {
public:
    Pika() { report(pika_renderer_new(&_renderer), "opening a GPU"); }
    ~Pika() {
        if (_renderer)
            report(pika_renderer_free(_renderer), "shutting down");
    }
    Pika(Pika const &) = delete;
    Pika &operator=(Pika const &) = delete;

    // The renderer to call, or null when there is nothing to render with:
    // no GPU opened, or Pika panicked, after which it renders nothing until
    // the delegate is recreated.
    pika_renderer *renderer() const { return _stopped ? nullptr : _renderer; }

    // Reports a failed call, saying what the delegate was doing. After a
    // panic every call fails, so the first to see it stops the rest and is
    // the only one reported.
    void report(int status, std::string const &what) const {
        if (status == PIKA_OK || _stopped)
            return;
        if (status == PIKA_PANIC && _stopped.exchange(true))
            return;
        fail(what + ": " + pika_last_error());
    }

private:
    pika_renderer *_renderer = nullptr;
    // Hydra reads the stats through a const delegate.
    mutable std::atomic<bool> _stopped = false;
};

PXR_NAMESPACE_CLOSE_SCOPE
