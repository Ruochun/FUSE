#include <fuse/reconstruct.h>
#include <iostream>
#include <cmath>
int main() {
    try {
        const double v[] = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1};
        const std::uint64_t f[] = {0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4,
                                   3, 7, 6, 3, 6, 2, 0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5};
        fuse::ReconstructionOptions o;
        o.voxel_size = 0.1;
        auto out = fuse::reconstruct({{v, 8, f, 12}}, o);
        if (out.vertices.empty() || out.faces.empty() || out.peak_device_bytes > o.max_memory_bytes)
            return 1;
        double volume = 0;
        for (std::size_t i = 0; i < out.faces.size(); i += 3) {
            const auto* a = &out.vertices[3 * out.faces[i]];
            const auto* b = &out.vertices[3 * out.faces[i + 1]];
            const auto* c = &out.vertices[3 * out.faces[i + 2]];
            volume += (a[0] * (b[1] * c[2] - b[2] * c[1]) + a[1] * (b[2] * c[0] - b[0] * c[2]) +
                       a[2] * (b[0] * c[1] - b[1] * c[0])) /
                      6;
        }
        if (std::abs(volume - 1) > 0.04) {
            std::cerr << "cube volume " << volume << '\n';
            return 1;
        }
        std::cout << "Native reconstruction passed; cube volume " << volume << '\n';
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
