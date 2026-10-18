#pragma once

#include <vector>

namespace fuse {
// M0 diagnostic, not a reconstruction API. Returns NanoVDB samples read on CUDA.
// Throws std::invalid_argument for invalid devices and std::runtime_error for
// CUDA failures. Calls are synchronous; the caller's current device is restored.
std::vector<float> smoke_test(int device = 0);
const char* nanovdb_revision() noexcept;
const char* cuda_architectures() noexcept;
int cuda_build_version() noexcept;
}  // namespace fuse
