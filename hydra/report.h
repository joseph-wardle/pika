// How the delegate reports errors to the host.
#pragma once

#include "pika.h"

#include <pxr/base/tf/diagnostic.h>

#include <cstdio>
#include <string>

PXR_NAMESPACE_OPEN_SCOPE

// husk drops every Tf diagnostic, so errors also go to stderr.
inline void fail(std::string const &message) {
    fprintf(stderr, "pika: error: %s\n", message.c_str());
    TF_RUNTIME_ERROR("Pika: %s", message.c_str());
}

// Reports a failed Pika call, saying what the delegate was doing.
inline void report(int status, std::string const &what) {
    if (status != PIKA_OK)
        fail(what + ": " + pika_last_error());
}

PXR_NAMESPACE_CLOSE_SCOPE
