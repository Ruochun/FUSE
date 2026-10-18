#pragma once
#include "geometry/scene.h"

namespace fuse::detail {
// Cube face orders use the same increasing in-plane axes on both incident cells.
// Face ambiguity uses the bilinear asymptotic decider. Interior ambiguity is
// resolved into separate contour disks, one center vertex per loop (not MC33).
struct Cell {
    double values[8];
    int adjacent[12][2];
    int face_for_arc[12][2];
    int degree[12];
    FUSE_HD static Vec corner(int c) {
        const int coords[8][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                                  {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
        return {double(coords[c][0]), double(coords[c][1]), double(coords[c][2])};
    }
    FUSE_HD static void endpoints(int e, int& a, int& b) {
        const int edges[12][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6},
                                  {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
        a = edges[e][0];
        b = edges[e][1];
    }
    FUSE_HD void connect(int a, int b, int face) {
        adjacent[a][degree[a]] = b;
        face_for_arc[a][degree[a]++] = face;
        adjacent[b][degree[b]] = a;
        face_for_arc[b][degree[b]++] = face;
    }
    FUSE_HD void build() {
        for (int e = 0; e < 12; ++e)
            degree[e] = 0;
        const int fc[6][4] = {{0, 1, 2, 3}, {4, 5, 6, 7}, {0, 1, 5, 4}, {3, 2, 6, 7}, {0, 3, 7, 4}, {1, 2, 6, 5}};
        const int fe[6][4] = {{0, 1, 2, 3}, {4, 5, 6, 7}, {0, 9, 4, 8}, {2, 10, 6, 11}, {3, 11, 7, 8}, {1, 10, 5, 9}};
        for (int f = 0; f < 6; ++f) {
            int crossing[4], count = 0;
            for (int e = 0; e < 4; ++e)
                if ((values[fc[f][e]] < 0) != (values[fc[f][(e + 1) % 4]] < 0))
                    crossing[count++] = fe[f][e];
            if (count == 2)
                connect(crossing[0], crossing[1], f);
            if (count == 4) {
                double q = values[fc[f][0]] * values[fc[f][2]] - values[fc[f][1]] * values[fc[f][3]];
                if (q >= 0) {
                    connect(fe[f][0], fe[f][1], f);
                    connect(fe[f][2], fe[f][3], f);
                } else {
                    connect(fe[f][3], fe[f][0], f);
                    connect(fe[f][1], fe[f][2], f);
                }
            }
        }
    }
    FUSE_HD int loop(int start, int* edges, unsigned& seen) const {
        int count = 0, previous = -1, current = start;
        do {
            edges[count++] = current;
            seen |= 1u << current;
            int next = adjacent[current][0] == previous ? adjacent[current][1] : adjacent[current][0];
            previous = current;
            current = next;
        } while (current != start && count < 12);
        return count;
    }
    FUSE_HD Vec position(int e) const {
        int a, b;
        endpoints(e, a, b);
        double t = values[a] / (values[a] - values[b]);
        return corner(a) + (corner(b) - corner(a)) * t;
    }
    FUSE_HD Vec gradient(Vec p) const {
        Vec g{0, 0, 0};
        for (int i = 0; i < 8; ++i) {
            Vec c = corner(i);
            double x = c.x ? p.x : 1 - p.x, y = c.y ? p.y : 1 - p.y, z = c.z ? p.z : 1 - p.z;
            g.x += values[i] * (c.x ? 1 : -1) * y * z;
            g.y += values[i] * (c.y ? 1 : -1) * x * z;
            g.z += values[i] * (c.z ? 1 : -1) * x * y;
        }
        return g;
    }
    FUSE_HD bool reverse_loop(int first, int second) const {
        int f = face_for_arc[first][adjacent[first][0] == second ? 0 : 1];
        const Vec outward[6] = {{0, 0, -1}, {0, 0, 1}, {0, -1, 0}, {0, 1, 0}, {-1, 0, 0}, {1, 0, 0}};
        Vec a = position(first), b = position(second);
        // Boundary orientation: triangle (a,b,interior) must point toward +field.
        return dot(cross(b - a, outward[f] * -1.0), gradient((a + b) * 0.5)) < 0;
    }
};
}  // namespace fuse::detail
