// Standalone experimental helper, not linked into the FUSE wheel.
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/boost/graph/IO/polygon_mesh_io.h>
#include <CGAL/Polygon_mesh_processing/region_growing.h>
#include <CGAL/Polygon_mesh_processing/remesh_planar_patches.h>
#include <CGAL/Polygon_mesh_processing/manifoldness.h>
#include <CGAL/Polygon_mesh_processing/repair_degeneracies.h>
#include <CGAL/Polygon_mesh_processing/connected_components.h>
#include <CGAL/version.h>
#include <boost/property_map/vector_property_map.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>
#include <algorithm>
#define FUSE_STRINGIFY_IMPL(x) #x
#define FUSE_STRINGIFY(x) FUSE_STRINGIFY_IMPL(x)

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using Mesh = CGAL::Surface_mesh<K::Point_3>;
namespace PMP = CGAL::Polygon_mesh_processing;
using Clock = std::chrono::steady_clock;

auto triangle(const Mesh& mesh, Mesh::Face_index f) {
    auto h = mesh.halfedge(f);
    return std::array<K::Point_3, 3>{mesh.point(mesh.source(h)), mesh.point(mesh.target(h)),
                                     mesh.point(mesh.target(mesh.next(h)))};
}
void validate(const Mesh& mesh) {
    if (!CGAL::is_valid_polygon_mesh(mesh) || !CGAL::is_triangle_mesh(mesh))
        throw std::runtime_error("mesh must have valid oriented triangular connectivity");
    std::vector<Mesh::Halfedge_index> bad_vertices;
    PMP::non_manifold_vertices(mesh, std::back_inserter(bad_vertices));
    if (!bad_vertices.empty())
        throw std::runtime_error("non-manifold vertices are not supported; no repair was applied");
    for (auto f : mesh.faces()) {
        auto t = triangle(mesh, f);
        auto n = CGAL::cross_product(t[1] - t[0], t[2] - t[0]);
        if (!(n.squared_length() > 0) || !std::isfinite(n.squared_length()))
            throw std::runtime_error("mesh has degenerate or nonfinite triangles");
    }
}
std::size_t components(Mesh& mesh) {
    auto map = mesh.add_property_map<Mesh::Face_index, std::size_t>("f:cc", 0).first;
    auto count = PMP::connected_components(mesh, map);
    mesh.remove_property_map(map);
    return count;
}
long long euler(const Mesh& mesh) {
    return static_cast<long long>(mesh.number_of_vertices()) - static_cast<long long>(mesh.number_of_edges()) +
           static_cast<long long>(mesh.number_of_faces());
}
// Shared interfaces are processed once. Split long chains into bounded blocks
// before geometric subdivision, so a long straight boundary cannot trigger
// unbounded repeated least-squares fitting or quadratic work.
std::size_t detect_boundary_corners(const Mesh& mesh,
                                    const std::vector<std::size_t>& regions,
                                    std::size_t nregions,
                                    std::vector<std::size_t>& corners,
                                    std::vector<bool>& constraints,
                                    double cosine,
                                    double distance) {
    const auto none = std::size_t(-1);
    std::vector<std::size_t> counts(nregions, 0);
    for (auto f : mesh.faces())
        ++counts[regions[f]];
    std::vector<unsigned char> degree(mesh.number_of_vertices(), 0);
    std::size_t ncorners = 0;
    auto keep = [&](Mesh::Vertex_index v) {
        if (corners[v] == none)
            corners[v] = ncorners++;
    };
    for (auto e : mesh.edges()) {
        auto h = mesh.halfedge(e);
        auto a = mesh.face(h), b = mesh.face(mesh.opposite(h));
        auto ra = a == Mesh::null_face() ? none : regions[a], rb = b == Mesh::null_face() ? none : regions[b];
        if (ra == rb)
            continue;
        constraints[e] = true;
        for (auto v : {mesh.source(h), mesh.target(h)}) {
            if (degree[v] < 3)
                ++degree[v];
            if ((ra != none && counts[ra] == 1) || (rb != none && counts[rb] == 1))
                keep(v);
        }
    }
    for (auto v : mesh.vertices())
        if (degree[v] && degree[v] != 2)
            keep(v);
    std::vector<bool> visited(mesh.number_of_edges(), false);
    auto handle = [&](Mesh::Halfedge_index h) {
        std::vector<Mesh::Vertex_index> chain{mesh.source(h)};
        while (true) {
            visited[mesh.edge(h)] = true;
            auto v = mesh.target(h);
            chain.push_back(v);
            if (corners[v] != none)
                break;
            auto next = Mesh::null_halfedge();
            for (auto ih : CGAL::halfedges_around_target(mesh.halfedge(v), mesh)) {
                if (constraints[mesh.edge(ih)] && mesh.edge(ih) != mesh.edge(h)) {
                    next = mesh.opposite(ih);
                    break;
                }
            }
            if (next == Mesh::null_halfedge() || visited[mesh.edge(next)]) {
                keep(v);
                break;
            }
            h = next;
        }
        keep(chain.front());
        keep(chain.back());
        std::vector<std::pair<std::size_t, std::size_t>> stack{{0, chain.size() - 1}};
        while (!stack.empty()) {
            auto [first, last] = stack.back();
            stack.pop_back();
            if (last - first <= 1)
                continue;
            std::size_t split = none;
            auto a = mesh.point(chain[first]), b = mesh.point(chain[last]);
            auto chord = b - a;
            double length2 = chord.squared_length(), worst = 1;
            if (last - first > 256 || !(length2 > 0))
                split = (first + last) / 2;
            else {
                K::Segment_3 segment(a, b);
                for (auto i = first; i < last; ++i) {
                    auto p = mesh.point(chain[i + 1]);
                    auto delta = p - mesh.point(chain[i]);
                    double c = (delta * chord) / std::sqrt(delta.squared_length() * length2);
                    double error = (1 - std::clamp(c, -1.0, 1.0)) / std::max(1 - cosine, 1e-30);
                    if (!std::isfinite(c))
                        error = std::numeric_limits<double>::infinity();
                    if (error > worst) {
                        worst = error;
                        split = i + 1 < last ? i + 1 : i;
                    }
                    if (i + 1 < last) {
                        error = CGAL::squared_distance(p, segment) / (distance * distance);
                        if (error > worst) {
                            worst = error;
                            split = i + 1;
                        }
                    }
                }
            }
            if (split != none) {
                if (split <= first || split >= last)
                    split = (first + last) / 2;
                keep(chain[split]);
                stack.emplace_back(first, split);
                stack.emplace_back(split, last);
            }
        }
    };
    for (auto v : mesh.vertices())
        if (corners[v] != none) {
            for (auto ih : CGAL::halfedges_around_target(mesh.halfedge(v), mesh)) {
                auto h = mesh.opposite(ih);
                if (constraints[mesh.edge(h)] && !visited[mesh.edge(h)])
                    handle(h);
            }
        }
    // Anchor and split closed boundary cycles (including hole boundaries).
    for (auto e : mesh.edges())
        if (constraints[e] && !visited[e]) {
            auto h = mesh.halfedge(e);
            keep(mesh.source(h));
            handle(h);
        }
    return ncorners;
}

// Verify complete boundary chains against their replacement chord. Local
// line-region decisions alone must not accumulate drift around a curved hole.
std::size_t protect_boundary_chains(const Mesh& mesh,
                                    const std::vector<bool>& constraints,
                                    std::vector<std::size_t>& corners,
                                    std::size_t& ncorners,
                                    double cosine,
                                    double distance) {
    const auto none = std::size_t(-1);
    std::vector<bool> visited(mesh.number_of_edges(), false);
    std::size_t protected_vertices = 0;
    auto protect = [&](Mesh::Vertex_index v) {
        if (corners[v] == none) {
            corners[v] = ncorners++;
            ++protected_vertices;
        }
    };
    for (auto start : mesh.vertices()) {
        if (corners[start] == none)
            continue;
        for (auto incoming : CGAL::halfedges_around_target(mesh.halfedge(start), mesh)) {
            auto h = mesh.opposite(incoming);
            if (!constraints[mesh.edge(h)] || visited[mesh.edge(h)])
                continue;
            std::vector<Mesh::Vertex_index> chain{start};
            while (true) {
                visited[mesh.edge(h)] = true;
                auto v = mesh.target(h);
                chain.push_back(v);
                if (corners[v] != none)
                    break;
                auto next = Mesh::null_halfedge();
                std::size_t degree = 0;
                for (auto ih : CGAL::halfedges_around_target(mesh.halfedge(v), mesh)) {
                    if (!constraints[mesh.edge(ih)])
                        continue;
                    ++degree;
                    if (mesh.edge(ih) != mesh.edge(h))
                        next = mesh.opposite(ih);
                }
                if (degree != 2 || next == Mesh::null_halfedge() || visited[mesh.edge(next)]) {
                    protect(v);
                    break;
                }
                h = next;
            }
            auto a = mesh.point(chain.front()), b = mesh.point(chain.back());
            auto chord = b - a;
            const double length2 = chord.squared_length();
            bool valid = length2 > 0;
            if (valid) {
                K::Segment_3 segment(a, b);
                for (std::size_t i = 1; i < chain.size(); ++i) {
                    auto p = mesh.point(chain[i]);
                    auto delta = p - mesh.point(chain[i - 1]);
                    double c = (delta * chord) / std::sqrt(delta.squared_length() * length2);
                    if (!std::isfinite(c) || c < cosine || CGAL::squared_distance(p, segment) > distance * distance) {
                        valid = false;
                        break;
                    }
                }
            }
            if (!valid)
                for (auto v : chain)
                    protect(v);
        }
    }
    // A boundary cycle without an identified corner is kept in full.
    for (auto e : mesh.edges())
        if (constraints[e] && !visited[e]) {
            auto h = mesh.halfedge(e);
            protect(mesh.source(h));
            protect(mesh.target(h));
        }
    return protected_vertices;
}
int main(int argc, char** argv) {
    try {
        if (argc != 6)
            throw std::runtime_error("usage: fuse_planar input output angle_degrees plane_distance stats.json");
        const double angle = std::stod(argv[3]), distance = std::stod(argv[4]);
        if (!(angle > 0 && angle < 90) || !std::isfinite(distance) || distance <= 0)
            throw std::runtime_error("invalid tolerances");
        const double cosine = std::cos(angle * std::acos(-1.0) / 180);
        auto start = Clock::now();
        auto progress = [&](const char* stage) {
            std::cout << stage << " elapsed_s=" << std::chrono::duration<double>(Clock::now() - start).count()
                      << std::endl;
        };
        Mesh mesh;
        progress("load");
        if (!CGAL::IO::read_polygon_mesh(argv[1], mesh) || mesh.is_empty())
            throw std::runtime_error("cannot read nonempty manifold mesh");
        validate(mesh);
        const auto in_v = mesh.number_of_vertices(), in_f = mesh.number_of_faces();
        const auto in_components = components(mesh);
        const auto in_euler = euler(mesh);
        progress("detect planar regions");
        std::vector<std::size_t> regions(in_f);
        auto region_map = CGAL::make_random_access_property_map(regions);
        boost::vector_property_map<K::Plane_3> planes;
        auto nregions = PMP::region_growing_of_planes_on_faces(
            mesh, region_map,
            CGAL::parameters::cosine_of_maximum_angle(cosine).maximum_distance(distance).region_primitive_map(planes));
        progress("audit final fitted planes");
        // The evolving fit can drift. Check EVERY member against the final plane;
        // leave a rejected region's triangles individually unchanged.
        std::vector<bool> rejected(nregions, false);
        std::vector<double> max_distance(nregions, 0), min_cosine(nregions, 1);
        for (auto f : mesh.faces()) {
            auto id = regions[f];
            if (id >= nregions)
                throw std::runtime_error("region growing left an unassigned face");
            auto t = triangle(mesh, f);
            const auto& plane = get(planes, id);
            auto n = CGAL::cross_product(t[1] - t[0], t[2] - t[0]);
            auto pn = plane.orthogonal_vector();
            double c = (n * pn) / std::sqrt(n.squared_length() * pn.squared_length());
            double d = 0;
            for (const auto& p : t)
                d = std::max(d, std::sqrt(CGAL::squared_distance(p, plane)));
            max_distance[id] = std::max(max_distance[id], d);
            min_cosine[id] = std::min(min_cosine[id], c);
            if (!std::isfinite(c) || !std::isfinite(d) || c < cosine || d > distance)
                rejected[id] = true;
        }
        std::vector<std::size_t> remap(nregions, std::size_t(-1));
        std::vector<K::Vector_3> normals;
        std::size_t rejected_regions = 0, fallback_faces = 0;
        double retained_distance = 0, retained_angle = 0;
        for (std::size_t id = 0; id < nregions; ++id) {
            if (rejected[id]) {
                ++rejected_regions;
                continue;
            }
            remap[id] = normals.size();
            normals.push_back(get(planes, id).orthogonal_vector());
            retained_distance = std::max(retained_distance, max_distance[id]);
            retained_angle =
                std::max(retained_angle, std::acos(std::clamp(min_cosine[id], -1.0, 1.0)) * 180 / std::acos(-1.0));
        }
        for (auto f : mesh.faces()) {
            auto id = regions[f];
            if (!rejected[id])
                regions[f] = remap[id];
            else {
                auto t = triangle(mesh, f);
                regions[f] = normals.size();
                normals.push_back(CGAL::cross_product(t[1] - t[0], t[2] - t[0]));
                ++fallback_faces;
            }
        }
        // Release temporary per-region data before allocating corner structures.
        std::vector<std::size_t>().swap(remap);
        std::vector<double>().swap(max_distance);
        std::vector<double>().swap(min_cosine);
        std::cout << "detected_regions=" << nregions << " final_regions=" << normals.size()
                  << " fallback_faces=" << fallback_faces << std::endl;
        progress("detect shared patch boundaries and corners");
        std::vector<std::size_t> corners(in_v, std::size_t(-1));
        std::vector<bool> constraints(mesh.number_of_edges(), false);
        auto corner_map = CGAL::make_random_access_property_map(corners);
        auto edge_map = CGAL::make_random_access_property_map(constraints);
        auto ncorners = detect_boundary_corners(mesh, regions, normals.size(), corners, constraints, cosine, distance);
        progress("audit shared boundary chains");
        auto protected_boundary_vertices =
            protect_boundary_chains(mesh, constraints, corners, ncorners, cosine, distance);
        progress("retriangulate patches");
        Mesh out;
        bool remeshed = PMP::remesh_almost_planar_patches(
            mesh, out, normals.size(), ncorners, region_map, corner_map, edge_map,
            CGAL::parameters::patch_normal_map(CGAL::make_random_access_property_map(normals)));
        // CGAL copies original triangles for patches it cannot retriangulate.
        // A false return can therefore still yield a valid partially reduced mesh.
        // Invalid polygon soup leaves the output empty and is always rejected.
        if (out.is_empty())
            throw std::runtime_error("CGAL did not produce a valid output mesh");
        progress("validate output");
        validate(out);
        auto out_components = components(out);
        auto out_euler = euler(out);
        if (in_components != out_components || in_euler != out_euler || CGAL::is_closed(mesh) != CGAL::is_closed(out))
            throw std::runtime_error("output changed component count, Euler characteristic, or closedness");
        if (out.number_of_faces() > in_f)
            throw std::runtime_error("remeshing increased face count");
        progress("save");
        if (!CGAL::IO::write_polygon_mesh(argv[2], out, CGAL::parameters::stream_precision(17).use_binary_mode(true)))
            throw std::runtime_error("could not save output mesh");
        std::ofstream stats(argv[5]);
        stats << std::setprecision(17) << "{\n\"cgal_version\":\"" << FUSE_STRINGIFY(CGAL_VERSION) << "\",\n"
              << "\"input_vertices\":" << in_v << ",\"input_faces\":" << in_f << ",\n"
              << "\"output_vertices\":" << out.number_of_vertices() << ",\"output_faces\":" << out.number_of_faces()
              << ",\n"
              << "\"all_patches_retriangulated\":" << (remeshed ? "true" : "false") << ",\n"
              << "\"components\":" << out_components << ",\"euler_characteristic\":" << out_euler << ",\n"
              << "\"protected_boundary_vertices\":" << protected_boundary_vertices << ",\n"
              << "\"detected_regions\":" << nregions << ",\"final_regions\":" << normals.size() << ",\n"
              << "\"rejected_regions\":" << rejected_regions << ",\"fallback_faces\":" << fallback_faces << ",\n"
              << "\"max_retained_plane_distance\":" << retained_distance
              << ",\"max_retained_normal_angle_degrees\":" << retained_angle << ",\n"
              << "\"seconds\":" << std::chrono::duration<double>(Clock::now() - start).count() << "\n}\n";
        if (!stats)
            throw std::runtime_error("could not save statistics");
        progress("done");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
