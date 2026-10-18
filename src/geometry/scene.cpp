#include "scene.h"
#include <numeric>
#include <string>
#include <unordered_map>

namespace fuse::detail {
namespace {
struct Edge {
    std::uint64_t a, b;
    bool operator==(const Edge& other) const { return a == other.a && b == other.b; }
};
struct Hash {
    std::size_t operator()(Edge e) const {
        return std::hash<std::uint64_t>{}(e.a) ^
               (std::hash<std::uint64_t>{}(e.b) + 0x9e3779b9 + (e.a << 6) + (e.a >> 2));
    }
};
struct Incidence {
    std::size_t face;
    int edge;
    bool forward;
    unsigned count;
};
Vec vertex(const MeshView& m, std::uint64_t i) {
    return {m.vertices[3 * i], m.vertices[3 * i + 1], m.vertices[3 * i + 2]};
}
void topology(const MeshView& m) {
    std::unordered_map<Edge, Incidence, Hash> edges;
    std::vector<std::size_t> adjacent(m.face_count * 3, SIZE_MAX);
    std::vector<std::size_t> first(m.vertex_count, SIZE_MAX), degree(m.vertex_count, 0);
    for (std::size_t f = 0; f < m.face_count; ++f) {
        for (int k = 0; k < 3; ++k) {
            auto a = m.faces[3 * f + k], b = m.faces[3 * f + (k + 1) % 3];
            first[a] = f;
            ++degree[a];
            Edge key{std::min(a, b), std::max(a, b)};
            auto [it, inserted] = edges.emplace(key, Incidence{f, k, a < b, 1});
            if (!inserted) {
                auto& old = it->second;
                if (++old.count != 2)
                    throw std::invalid_argument("mesh is nonmanifold: an edge has more than two faces");
                if (old.forward == (a < b))
                    throw std::invalid_argument("mesh has inconsistent face orientation");
                adjacent[3 * f + k] = old.face;
                adjacent[3 * old.face + old.edge] = f;
            }
        }
    }
    for (auto& entry : edges)
        if (entry.second.count != 2)
            throw std::invalid_argument("mesh is open: each edge must have two incident faces");
    // A pair of otherwise closed shells sharing only one vertex is not a manifold.
    for (std::size_t v = 0; v < m.vertex_count; ++v) {
        if (!degree[v])
            continue;
        auto current = first[v], previous = SIZE_MAX;
        std::size_t visited = 0;
        do {
            int k = 0;
            while (k < 3 && m.faces[current * 3 + k] != v)
                ++k;
            if (k == 3 || ++visited > degree[v])
                throw std::invalid_argument("mesh has a nonmanifold vertex link");
            auto a = adjacent[3 * current + k], b = adjacent[3 * current + (k + 2) % 3];
            auto next = a == previous ? b : a;
            previous = current;
            current = next;
        } while (current != first[v]);
        if (visited != degree[v])
            throw std::invalid_argument("mesh has a disconnected vertex link");
    }
}
Box bounds(const Triangle& t) {
    Box b;
    b.add(t.a);
    b.add(t.b);
    b.add(t.c);
    return b;
}
std::uint32_t build(Scene& s, std::uint32_t begin, std::uint32_t end) {
    const auto id = static_cast<std::uint32_t>(s.nodes.size());
    s.nodes.emplace_back();
    Box box;
    for (auto i = begin; i < end; ++i)
        box.add(bounds(s.triangles[i]));
    s.nodes[id].bounds = box;
    if (end - begin <= 4) {
        s.nodes[id].begin = begin;
        s.nodes[id].count = end - begin;
    } else {
        auto extent = box.hi - box.lo;
        int axis = extent.y > extent.x ? 1 : 0;
        if (extent.z > extent[axis])
            axis = 2;
        const auto mid = begin + (end - begin) / 2;
        std::nth_element(s.triangles.begin() + begin, s.triangles.begin() + mid, s.triangles.begin() + end,
                         [axis](const Triangle& a, const Triangle& b) {
                             return a.a[axis] + a.b[axis] + a.c[axis] < b.a[axis] + b.b[axis] + b.c[axis];
                         });
        build(s, begin, mid);
        build(s, mid, end);
    }
    s.nodes[id].escape = static_cast<std::uint32_t>(s.nodes.size());
    return id;
}
std::uint32_t build_shapes(Scene& s, std::uint32_t begin, std::uint32_t end) {
    const auto id = static_cast<std::uint32_t>(s.shape_nodes.size());
    s.shape_nodes.emplace_back();
    Box bounds;
    for (auto i = begin; i < end; ++i)
        bounds.add(s.shapes[i].bounds);
    s.shape_nodes[id].bounds = bounds;
    if (end - begin <= 4) {
        s.shape_nodes[id].begin = begin;
        s.shape_nodes[id].count = end - begin;
    } else {
        const auto extent = bounds.hi - bounds.lo;
        int axis = extent.y > extent.x ? 1 : 0;
        if (extent.z > extent[axis])
            axis = 2;
        const auto mid = begin + (end - begin) / 2;
        std::nth_element(s.shapes.begin() + begin, s.shapes.begin() + mid, s.shapes.begin() + end,
                         [axis](const Shape& a, const Shape& b) {
                             return a.bounds.lo[axis] + a.bounds.hi[axis] < b.bounds.lo[axis] + b.bounds.hi[axis];
                         });
        build_shapes(s, begin, mid);
        build_shapes(s, mid, end);
    }
    s.shape_nodes[id].escape = static_cast<std::uint32_t>(s.shape_nodes.size());
    return id;
}
}  // namespace
Scene prepare(const std::vector<MeshView>& meshes, const ReconstructionOptions& options) {
    Scene s;
    Box world;
    std::uint64_t triangles = 0;
    for (std::size_t mesh_id = 0; mesh_id < meshes.size(); ++mesh_id) {
        const auto& m = meshes[mesh_id];
        try {
            if ((m.vertex_count && !m.vertices) || (m.face_count && !m.faces))
                throw std::invalid_argument("null mesh buffer");
            if (m.vertex_count > SIZE_MAX / 3 || m.face_count > SIZE_MAX / 3)
                throw ResourceError("mesh index arithmetic exceeds address range");
            if (m.face_count > UINT32_MAX / 2 || triangles + m.face_count > UINT32_MAX / 2)
                throw ResourceError("too many triangles for the BVH");
            triangles += m.face_count;
            for (std::size_t i = 0; i < m.vertex_count * 3; ++i)
                if (!std::isfinite(m.vertices[i]))
                    throw std::invalid_argument("vertices must be finite");
            for (std::size_t i = 0; i < m.face_count; ++i) {
                auto a = m.faces[3 * i], b = m.faces[3 * i + 1], c = m.faces[3 * i + 2];
                if (a >= m.vertex_count || b >= m.vertex_count || c >= m.vertex_count)
                    throw std::invalid_argument("face index is outside the vertex array");
                Vec va = vertex(m, a), vb = vertex(m, b), vc = vertex(m, c);
                Vec ab = (vb - va) * (1.0 / options.voxel_size), ac = (vc - va) * (1.0 / options.voxel_size);
                auto n = cross(ab, ac);
                double area2 = dot(n, n);
                if (!(area2 > 0) || !std::isfinite(area2))
                    throw std::invalid_argument("triangle has zero area or exceeds numeric range at this voxel_size");
                world.add(va);
                world.add(vb);
                world.add(vc);
            }
            if (options.validate_topology && m.face_count)
                topology(m);
        } catch (const std::invalid_argument& e) {
            throw std::invalid_argument("mesh " + std::to_string(mesh_id) + ": " + e.what());
        }
    }
    if (!triangles)
        return s;
    std::uint64_t samples = 1;
    for (int a = 0; a < 3; ++a) {
        const double span = (world.hi[a] - world.lo[a]) / options.voxel_size;
        const double count = std::ceil(span) + 6;
        if (!std::isfinite(count) || count > INT32_MAX - 16 || count < 2)
            throw ResourceError("grid dimensions exceed coordinate limits; increase voxel_size");
        s.dimensions[a] = static_cast<std::uint32_t>(count);
        if (samples > (options.sparse ? UINT64_MAX / 3 : std::uint64_t(INT32_MAX / 3)) / s.dimensions[a])
            throw ResourceError("grid sample count exceeds indexing limits; increase voxel_size");
        samples *= s.dimensions[a];
        s.origin[a] = world.lo[a] - 2.5 * options.voxel_size;
        if (!std::isfinite(s.origin[a]) || s.origin[a] + options.voxel_size == s.origin[a])
            throw std::invalid_argument("voxel_size cannot be represented at this world coordinate magnitude");
    }
    // Preflight before building/uploading BVHs; actual allocations are budgeted too.
    if (!options.sparse && samples > options.max_memory_bytes / 48)
        throw ResourceError("dense grid exceeds max_memory_bytes; increase voxel_size or the memory budget");
    s.triangles.reserve(triangles);
    for (const auto& m : meshes) {
        if (!m.face_count)
            continue;
        const auto begin = static_cast<std::uint32_t>(s.triangles.size());
        const auto local = [&](std::uint64_t i) {
            auto v = vertex(m, i);
            // Subtract the scene minimum before scaling; retain the host world transform.
            return Vec{(v.x - world.lo.x) / options.voxel_size + 2.5, (v.y - world.lo.y) / options.voxel_size + 2.5,
                       (v.z - world.lo.z) / options.voxel_size + 2.5};
        };
        for (std::size_t i = 0; i < m.face_count; ++i) {
            Triangle t{local(m.faces[3 * i]), local(m.faces[3 * i + 1]), local(m.faces[3 * i + 2])};
            Vec n = cross(t.b - t.a, t.c - t.a);
            if (!(dot(n, n) > 0) || !std::isfinite(dot(n, n)))
                throw std::invalid_argument("triangle collapses in grid coordinates; adjust voxel_size or scene scale");
            s.triangles.push_back(t);
        }
        auto root = build(s, begin, static_cast<std::uint32_t>(s.triangles.size()));
        s.shapes.push_back({s.nodes[root].bounds, root, s.nodes[root].escape});
    }
    if (s.shapes.size() > 4)
        build_shapes(s, 0, static_cast<std::uint32_t>(s.shapes.size()));
    return s;
}
}  // namespace fuse::detail
