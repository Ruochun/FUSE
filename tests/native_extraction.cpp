#include "geometry/scene.h"
#include <iostream>
#include <map>
#include <utility>

int main() {
    using namespace fuse;
    using namespace fuse::detail;
    try {
        // Pack every central cube sign configuration into a padded grid. Their
        // neighboring cells exercise transitions and all three shared-face axes.
        // Test equal magnitudes (exact saddle ties) and unequal magnitudes.
        for (int variant = 0; variant < 3; ++variant) {
            Scene scene;
            scene.dimensions = {256 * 20 + 2, 6, 6};
            const auto nx = scene.dimensions[0], ny = scene.dimensions[1], nz = scene.dimensions[2];
            std::vector<double> samples(std::size_t(nx) * ny * nz, 2.0);
            const int corners[8][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                                       {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
            for (int mask = 0; mask < 256; ++mask)
                for (int c = 0; c < 8; ++c) {
                    int x = mask * 20 + 1 + corners[c][0], y = 2 + corners[c][1], z = 2 + corners[c][2];
                    samples[(std::size_t(z) * ny + y) * nx + x] =
                        ((mask >> c) & 1 ? -1 : 1) * (variant ? 0.2 + (c * 7 + variant * 11) % 13 : 1.0);
                }
            // A plane covers the synthetic field's candidate blocks. This
            // exercises NanoVDB lookup across leaf/internal/root boundaries
            // (including x=4096) without coupling cases to mesh classification.
            double end = nx - 1;
            scene.triangles = {{{0, 0, 0}, {end, 0, 0}, {end, 5, 0}}, {{0, 0, 0}, {end, 5, 0}, {0, 5, 0}}};
            ReconstructionOptions options;
            options.voxel_size = 1;
            options.sparse = false;
            auto dense = run_cuda(scene, options, &samples);
            options.sparse = true;
            auto result = run_cuda(scene, options, &samples);
            if (result.vertices != dense.vertices || result.faces != dense.faces)
                throw std::runtime_error("sparse extraction differs from dense extraction");
            std::map<std::pair<std::uint64_t, std::uint64_t>, std::pair<int, int>> edges;
            for (std::size_t i = 0; i < result.faces.size(); i += 3)
                for (int k = 0; k < 3; ++k) {
                    auto a = result.faces[i + k], b = result.faces[i + (k + 1) % 3];
                    if (a >= result.vertices.size() / 3 || b >= result.vertices.size() / 3 || a == b)
                        throw std::runtime_error("invalid triangle indices");
                    auto& entry = edges[{std::min(a, b), std::max(a, b)}];
                    ++entry.first;
                    entry.second += a < b ? 1 : -1;
                }
            for (auto& entry : edges)
                if (entry.second.first != 2 || entry.second.second != 0)
                    throw std::runtime_error("GPU extraction has a seam or inconsistent winding");
        }
        std::cout << "768 configurations agree between sparse/dense GPU extraction with closed, consistently oriented "
                     "shared edges\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
