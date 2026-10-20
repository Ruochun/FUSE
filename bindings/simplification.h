// Compact CPU topology validation for simplification. No Python object graph.
#pragma once
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace fuse_simplification {
namespace nb = nanobind;
struct Edge {
    std::uint64_t key, corner;
};
struct Result {
    std::vector<std::uint32_t> faces, edges, adjacent;
    std::vector<std::size_t> offsets;
};
inline std::uint32_t root(std::vector<std::uint32_t>& parent, std::uint32_t i) {
    while (parent[i] != i) {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    return i;
}
inline void bind(nb::module_& m) {
    using V = nb::ndarray<nb::numpy, const double, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;
    using F = nb::ndarray<nb::numpy, const std::int64_t, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;
    m.def(
        "_inspect_mesh",
        [](V vertices, F faces, bool components_only) {
            auto result = std::make_unique<Result>();
            const auto n = faces.shape(0), nv = vertices.shape(0);
            if (n > std::numeric_limits<std::int32_t>::max() || nv > std::numeric_limits<std::int32_t>::max())
                throw std::invalid_argument("simplification backend requires 32-bit mesh sizes");
            const auto* f = faces.data();
            const auto* v = vertices.data();
            {
                nb::gil_scoped_release release;
                std::vector<Edge> records(3 * n);
                for (std::size_t i = 0; i < n; ++i) {
                    for (int j = 0; j < 3; ++j)
                        if (f[3 * i + j] < 0 || std::size_t(f[3 * i + j]) >= nv)
                            throw std::invalid_argument("face index is outside the vertex array");
                    double a[3], b[3], c[3];
                    for (int j = 0; j < 3; ++j) {
                        a[j] = v[3 * f[3 * i + 1] + j] - v[3 * f[3 * i] + j];
                        b[j] = v[3 * f[3 * i + 2] + j] - v[3 * f[3 * i] + j];
                    }
                    for (int j = 0; j < 3; ++j)
                        c[j] = a[(j + 1) % 3] * b[(j + 2) % 3] - a[(j + 2) % 3] * b[(j + 1) % 3];
                    double area = std::sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
                    if (!std::isfinite(area) || area <= 1e-28)
                        throw std::invalid_argument("mesh has degenerate triangles");
                    for (int j = 0; j < 3; ++j) {
                        auto u = std::uint32_t(f[3 * i + j]), w = std::uint32_t(f[3 * i + (j + 1) % 3]);
                        records[3 * i + j] = {(std::uint64_t(std::min(u, w)) << 32) | std::max(u, w), 3 * i + j};
                    }
                }
                std::sort(records.begin(), records.end(), [](const Edge& a, const Edge& b) { return a.key < b.key; });
                std::vector<std::uint64_t> twin(3 * n);
                std::vector<std::uint32_t> parent(n), sizes(n, 1);
                std::iota(parent.begin(), parent.end(), 0);
                if (!components_only) {
                    result->edges.reserve(3 * n);
                    result->adjacent.reserve(3 * n);
                }
                for (std::size_t i = 0; i < records.size();) {
                    auto end = i + 1;
                    while (end < records.size() && records[end].key == records[i].key)
                        ++end;
                    if (end - i != 2)
                        throw std::invalid_argument("mesh must be closed and edge-manifold");
                    auto a = records[i].corner, b = records[i + 1].corner;
                    if (f[3 * (a / 3) + (a % 3 + 2) % 3] == f[3 * (b / 3) + (b % 3 + 2) % 3])
                        throw std::invalid_argument("mesh has duplicate triangles");
                    if (f[a] == f[b])
                        throw std::invalid_argument("mesh winding must be consistent");
                    twin[a] = b;
                    twin[b] = a;
                    auto ra = root(parent, a / 3), rb = root(parent, b / 3);
                    if (ra != rb) {
                        if (sizes[ra] < sizes[rb])
                            std::swap(ra, rb);
                        parent[rb] = ra;
                        sizes[ra] += sizes[rb];
                    }
                    if (!components_only) {
                        result->edges.push_back(records[i].key >> 32);
                        result->edges.push_back(std::uint32_t(records[i].key));
                        result->adjacent.push_back(a / 3);
                        result->adjacent.push_back(b / 3);
                    }
                    i = end;
                }
                std::vector<Edge>().swap(records);
                // Rotation around a vertex: cross the outgoing edge, then take the
                // next outgoing edge. Exactly one cycle is required per used vertex.
                std::vector<bool> visited(3 * n, false), used(nv, false);
                for (std::size_t start = 0; start < 3 * n; ++start) {
                    if (visited[start])
                        continue;
                    if (used[f[start]])
                        throw std::invalid_argument("mesh must be vertex-manifold");
                    used[f[start]] = true;
                    auto c = start;
                    do {
                        visited[c] = true;
                        auto t = twin[c];
                        c = 3 * (t / 3) + (t % 3 + 1) % 3;
                    } while (c != start);
                }
                std::vector<std::uint64_t>().swap(twin);
                // Assign components in first-face order and keep each face list sorted.
                std::fill(sizes.begin(), sizes.end(), std::numeric_limits<std::uint32_t>::max());
                std::size_t count = 0;
                for (std::size_t i = 0; i < n; ++i) {
                    auto r = root(parent, i);
                    if (sizes[r] == std::numeric_limits<std::uint32_t>::max())
                        sizes[r] = count++;
                    parent[i] = r;
                }
                result->offsets.assign(count + 1, 0);
                for (auto r : parent)
                    ++result->offsets[sizes[r] + 1];
                for (std::size_t i = 1; i <= count; ++i)
                    result->offsets[i] += result->offsets[i - 1];
                auto cursor = result->offsets;
                result->faces.resize(n);
                for (std::size_t i = 0; i < n; ++i)
                    result->faces[cursor[sizes[parent[i]]]++] = i;
            }
            auto* p = result.get();
            nb::capsule owner(p, [](void* p) noexcept { delete static_cast<Result*>(p); });
            result.release();
            static std::uint32_t empty = 0;
            nb::list components;
            for (std::size_t i = 1; i < p->offsets.size(); ++i)
                components.append(nb::ndarray<nb::numpy, std::uint32_t>(p->faces.data() + p->offsets[i - 1],
                                                                        {p->offsets[i] - p->offsets[i - 1]}, owner));
            return nb::make_tuple(
                components,
                nb::ndarray<nb::numpy, std::uint32_t>(p->edges.empty() ? &empty : p->edges.data(),
                                                      {p->edges.size() / 2, 2}, owner),
                nb::ndarray<nb::numpy, std::uint32_t>(p->adjacent.empty() ? &empty : p->adjacent.data(),
                                                      {p->adjacent.size() / 2, 2}, owner));
        },
        nb::arg("vertices"), nb::arg("faces"), nb::arg("components_only") = false);
}
}  // namespace fuse_simplification
