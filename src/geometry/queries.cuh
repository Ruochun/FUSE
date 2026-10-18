#pragma once
#include "scene.h"

namespace fuse::detail {
// Closest point in triangle Voronoi regions; inputs have already passed the
// nonzero-area check. All arithmetic is double in shared grid coordinates.
__device__ inline double triangle_distance2(Vec p, const Triangle& t) {
    Vec ab=t.b-t.a, ac=t.c-t.a, ap=p-t.a;
    double d1=dot(ab,ap), d2=dot(ac,ap);
    if (d1<=0 && d2<=0) return dot(ap,ap);
    Vec bp=p-t.b;
    double d3=dot(ab,bp),d4=dot(ac,bp);
    if (d3>=0 && d4<=d3) return dot(bp,bp);
    double vc=d1*d4-d3*d2;
    if (vc<=0 && d1>=0 && d3<=0) {
        Vec q=ap-ab*(d1/(d1-d3)); return dot(q,q);
    }
    Vec cp=p-t.c;
    double d5=dot(ab,cp),d6=dot(ac,cp);
    if (d6>=0 && d5<=d6) return dot(cp,cp);
    double vb=d5*d2-d1*d6;
    if (vb<=0 && d2>=0 && d6<=0) {
        Vec q=ap-ac*(d2/(d2-d6)); return dot(q,q);
    }
    double va=d3*d6-d5*d4;
    if (va<=0 && d4-d3>=0 && d5-d6>=0) {
        Vec q=bp-(t.c-t.b)*((d4-d3)/((d4-d3)+(d5-d6))); return dot(q,q);
    }
    Vec n=cross(ab,ac);
    double projection=dot(ap,n);
    return projection*projection/dot(n,n);
}
__device__ inline double edge_yz(Vec a,Vec b,Vec p) {
    // Evaluate a shared edge in one canonical direction, then negate. Adjacent
    // triangles therefore obtain exactly opposite edge functions, even after rounding.
    bool reverse=a.y>b.y || (a.y==b.y && a.z>b.z);
    if (reverse) { Vec tmp=a; a=b; b=tmp; }
    double e=(b.y-a.y)*(p.z-a.z)-(b.z-a.z)*(p.y-a.y);
    return reverse ? -e:e;
}
__device__ inline bool half_open(double edge,Vec a,Vec b) {
    return edge>0 || (edge==0 && (b.z>a.z || (b.z==a.z && b.y<a.y)));
}
__device__ inline bool ray_hit(Vec p,Triangle t) {
    double area=edge_yz(t.a,t.b,t.c);
    if (area==0) return false; // ray parallel to this triangle
    if (area<0) { Vec tmp=t.b; t.b=t.c; t.c=tmp; area=-area; }
    double a=edge_yz(t.b,t.c,p), b=edge_yz(t.c,t.a,p), c=edge_yz(t.a,t.b,p);
    if (!half_open(a,t.b,t.c) || !half_open(b,t.c,t.a) || !half_open(c,t.a,t.b)) return false;
    // Relative x coordinates avoid cancellation when the hit lies near the sample.
    return a*(t.a.x-p.x)+b*(t.b.x-p.x)+c*(t.c.x-p.x)>0;
}
__device__ inline double signed_distance(Vec p,const Shape& shape,const Node* nodes,const Triangle* tris) {
    double distance=INFINITY;
    auto index=shape.root;
    while (index<shape.end) {
        const Node& n=nodes[index];
        if (n.bounds.distance2(p)>distance) { index=n.escape; continue; }
        for (std::uint32_t j=0;j<n.count;++j)
            distance=minval(distance,triangle_distance2(p,tris[n.begin+j]));
        ++index;
    }
    bool inside=false;
    if (shape.bounds.contains(p) && distance>1e-24) {
        index=shape.root;
        while (index<shape.end) {
            const Node& n=nodes[index];
            if (!n.bounds.ray(p)) { index=n.escape; continue; }
            for (std::uint32_t j=0;j<n.count;++j) inside ^= ray_hit(p,tris[n.begin+j]);
            ++index;
        }
    }
    double d=sqrt(maxval(distance,0.0));
    return inside ? -d:d;
}
} // namespace fuse::detail
