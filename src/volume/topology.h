#pragma once
#include "geometry/scene.h"

namespace fuse::detail {
// NanoVDB uint64 values are sample index + 1; zero means absent topology,
// never an exterior scalar value. Keys use the scene's global z,y,x order.
struct SparseTopology {
    std::vector<std::uint64_t> samples, cells;
    std::vector<unsigned char> grid;
    std::uint64_t blocks = 0;
};
SparseTopology make_topology(const Scene&, const ReconstructionOptions&);
}
