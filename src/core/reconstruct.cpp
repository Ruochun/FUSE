#include <fuse/reconstruct.h>
#include "geometry/scene.h"
#include <chrono>

namespace fuse {
ReconstructionResult reconstruct(const std::vector<MeshView>& meshes, const ReconstructionOptions& options) {
    const auto start = std::chrono::steady_clock::now();
    if (!std::isfinite(options.voxel_size) || options.voxel_size <= 0)
        throw std::invalid_argument("voxel_size must be finite and positive");
    if (options.device < 0)
        throw std::invalid_argument("device must be nonnegative");
    if (!options.max_memory_bytes)
        throw std::invalid_argument("max_memory_bytes must be positive");
    auto scene = detail::prepare(meshes, options);
    const auto prepared = std::chrono::steady_clock::now();
    ReconstructionResult result;
    if (!scene.shapes.empty())
        result = detail::run_cuda(scene, options);
    result.voxel_size = options.voxel_size;
    result.device = options.device;
    result.origin = scene.origin;
    result.grid_shape = scene.dimensions;
    result.prepare_ms = std::chrono::duration<double, std::milli>(prepared - start).count();
    return result;
}
}  // namespace fuse
