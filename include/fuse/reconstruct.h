#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace fuse {
struct MeshView {
    const double* vertices = nullptr; // interleaved xyz, world coordinates
    std::size_t vertex_count = 0;
    const std::uint64_t* faces = nullptr; // interleaved triangle indices
    std::size_t face_count = 0;
};
struct ReconstructionOptions {
    double voxel_size = 0.01;
    int device = 0;
    std::uint64_t max_memory_bytes = 512ull * 1024 * 1024;
    bool validate_topology = true;
    bool return_field = false;
};
class ResourceError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};
struct ReconstructionResult {
    std::vector<double> vertices; // xyz, world coordinates
    std::vector<std::uint64_t> faces; // triangle indices
    std::vector<double> field; // optional z,y,x samples, distances in world units
    std::array<double, 3> origin{}; // world position of sample (0,0,0)
    std::array<std::uint32_t, 3> grid_shape{}; // x,y,z sample counts
    double voxel_size = 0;
    int device = 0;
    std::uint64_t peak_device_bytes = 0; // FUSE allocations, excluding CUDA context
    double prepare_ms = 0, upload_ms = 0, field_ms = 0, extraction_ms = 0, download_ms = 0;
};
// Synchronous. Inputs remain valid and unchanged until return. No Python or
// upstream storage types cross this boundary. Empty scenes do not initialize CUDA.
ReconstructionResult reconstruct(const std::vector<MeshView>& meshes,
                                 const ReconstructionOptions& options = {});
} // namespace fuse
