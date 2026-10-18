#pragma once
#include <fuse/reconstruct.h>
#include <cmath>
#include <algorithm>
#include <limits>
#include <vector>

#ifdef __CUDACC__
#define FUSE_HD __host__ __device__
#else
#define FUSE_HD
#endif
namespace fuse::detail {
struct Vec {
    double x, y, z;
    FUSE_HD double operator[](int axis) const { return axis == 0 ? x : axis == 1 ? y : z; }
};
FUSE_HD inline Vec operator+(Vec a, Vec b) { return {a.x+b.x,a.y+b.y,a.z+b.z}; }
FUSE_HD inline Vec operator-(Vec a, Vec b) { return {a.x-b.x,a.y-b.y,a.z-b.z}; }
FUSE_HD inline Vec operator*(Vec a, double b) { return {a.x*b,a.y*b,a.z*b}; }
FUSE_HD inline double dot(Vec a, Vec b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
FUSE_HD inline Vec cross(Vec a, Vec b) { return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x}; }
FUSE_HD inline double minval(double a, double b) { return a < b ? a : b; }
FUSE_HD inline double maxval(double a, double b) { return a > b ? a : b; }
struct Box {
    Vec lo{INFINITY,INFINITY,INFINITY}, hi{-INFINITY,-INFINITY,-INFINITY};
    void add(Vec p) {
        lo = {std::min(lo.x,p.x),std::min(lo.y,p.y),std::min(lo.z,p.z)};
        hi = {std::max(hi.x,p.x),std::max(hi.y,p.y),std::max(hi.z,p.z)};
    }
    void add(const Box& b) { add(b.lo); add(b.hi); }
    FUSE_HD bool contains(Vec p) const {
        return p.x>=lo.x && p.x<=hi.x && p.y>=lo.y && p.y<=hi.y && p.z>=lo.z && p.z<=hi.z;
    }
    FUSE_HD double distance2(Vec p) const {
        double x=maxval(maxval(lo.x-p.x,0.0),p.x-hi.x);
        double y=maxval(maxval(lo.y-p.y,0.0),p.y-hi.y);
        double z=maxval(maxval(lo.z-p.z,0.0),p.z-hi.z);
        return x*x+y*y+z*z;
    }
    FUSE_HD bool ray(Vec p) const {
        return hi.x>p.x && lo.y<=p.y && hi.y>=p.y && lo.z<=p.z && hi.z>=p.z;
    }
};
struct Triangle { Vec a,b,c; };
struct Node { Box bounds; std::uint32_t begin=0, count=0, escape=0; };
struct Shape { Box bounds; std::uint32_t root=0, end=0; };
struct Scene {
    std::vector<Triangle> triangles;
    std::vector<Node> nodes;
    std::vector<Shape> shapes;
    std::array<double,3> origin{};
    std::array<std::uint32_t,3> dimensions{};
};
Scene prepare(const std::vector<MeshView>& meshes, const ReconstructionOptions& options);
// The optional field is an internal validation seam for testing extraction
// independently of mesh queries; it is not exposed by the public API. Sparse
// tests must provide triangles covering the supplied field's crossing cells.
ReconstructionResult run_cuda(const Scene& scene, const ReconstructionOptions& options,
                              const std::vector<double>* sampled_field = nullptr);
} // namespace fuse::detail
