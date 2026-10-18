#include "topology.h"
#include <nanovdb/tools/GridBuilder.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <unordered_set>
#include <cstring>
#include <chrono>

namespace fuse::detail {
namespace {
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point start) { return std::chrono::duration<double,std::milli>(Clock::now()-start).count(); }
struct Interval { std::uint64_t begin,end; };
// Sort/merge contiguous x-runs instead of sorting every cell/sample key.
// Emitted keys retain exactly M2's global z/y/x ordering and shared halos.
std::vector<std::uint64_t> expand(std::vector<Interval>& rows,std::uint64_t limit) {
    std::sort(rows.begin(),rows.end(),[](const Interval& a,const Interval& b){return a.begin<b.begin;});
    std::size_t count=0;
    for(auto row:rows) {
        if(count && row.begin<=rows[count-1].end+1) rows[count-1].end=std::max(rows[count-1].end,row.end);
        else rows[count++]=row;
    }
    std::uint64_t size=0;
    for(std::size_t i=0;i<count;++i) {
        const auto n=rows[i].end-rows[i].begin+1;
        if(n>limit-size) throw ResourceError("sparse active counts exceed scan limits or max_memory_bytes");
        size+=n;
    }
    std::vector<std::uint64_t> keys;keys.reserve(size);
    for(std::size_t i=0;i<count;++i)
        for(auto k=rows[i].begin;k<=rows[i].end;++k) keys.push_back(k);
    return keys;
}
struct Int3 { int v[3]; int& operator[](int a) { return v[a]; } int operator[](int a) const { return v[a]; } };
// Conservative triangle/box SAT. Projection margins cover arithmetic roundoff
// and the field's symbolic zero offset. Candidate discovery never uses signs.
bool overlaps(const Triangle& t, Int3 low, Int3 high) {
    Vec center{}, half{};
    center={4.0*(low[0]+high[0]+1),4.0*(low[1]+high[1]+1),4.0*(low[2]+high[2]+1)};
    half={4.0*(high[0]-low[0]+1),4.0*(high[1]-low[1]+1),4.0*(high[2]-low[2]+1)};
    Vec p[3]={t.a-center,t.b-center,t.c-center};
    double scale=1;
    for(auto v:{t.a,t.b,t.c,center}) for(int a=0;a<3;++a) scale=std::max(scale,std::abs(v[a]));
    const double margin=1e-7+128*std::numeric_limits<double>::epsilon()*scale;
    auto separated=[&](Vec axis) {
        double lo=dot(p[0],axis),hi=lo;
        for(int i=1;i<3;++i) { double d=dot(p[i],axis);lo=std::min(lo,d);hi=std::max(hi,d); }
        double r=std::abs(axis.x)*half.x+std::abs(axis.y)*half.y+std::abs(axis.z)*half.z;
        r+=margin*(std::abs(axis.x)+std::abs(axis.y)+std::abs(axis.z));
        return lo>r || hi<-r;
    };
    const Vec axes[]={{1,0,0},{0,1,0},{0,0,1}};
    for(auto axis:axes) if(separated(axis)) return false;
    Vec edges[]={t.b-t.a,t.c-t.b,t.a-t.c};
    if(separated(cross(edges[0],edges[1]))) return false;
    for(auto edge:edges) for(auto axis:axes) if(separated(cross(edge,axis))) return false;
    return true;
}
}
SparseTopology make_topology(const Scene& s,const ReconstructionOptions& options) {
    SparseTopology out;
    auto start=Clock::now();
    const auto nx=s.dimensions[0],ny=s.dimensions[1];
    auto key=[&](int x,int y,int z) { return (std::uint64_t(z)*ny+y)*nx+x; };
    std::unordered_set<std::uint64_t> blocks;
    // Bound host candidate growth before NanoVDB construction. This guard is
    // conservative (boundary blocks may be partial); exact device budgeting follows.
    const auto max_blocks=options.max_memory_bytes/(512*16);
    auto visit=[&](auto&& self,const Triangle& t,Int3 lo,Int3 hi)->void {
        if(!overlaps(t,lo,hi)) return;
        int axis=0;
        for(int a=1;a<3;++a) if(hi[a]-lo[a]>hi[axis]-lo[axis]) axis=a;
        if(lo[axis]==hi[axis]) {
            blocks.insert(key(8*lo[0],8*lo[1],8*lo[2]));
            if(blocks.size()>max_blocks) throw ResourceError("sparse candidate topology exceeds max_memory_bytes; increase voxel_size");
            return;
        }
        int mid=lo[axis]+(hi[axis]-lo[axis])/2;
        auto left=hi;left[axis]=mid;self(self,t,lo,left);
        auto right=lo;right[axis]=mid+1;self(self,t,right,hi);
    };
    for(const auto& t:s.triangles) {
        Int3 lo{},hi{};
        for(int a=0;a<3;++a) {
            double mn=std::min({t.a[a],t.b[a],t.c[a]}),mx=std::max({t.a[a],t.b[a],t.c[a]});
            double margin=1e-7+128*std::numeric_limits<double>::epsilon()*std::max(std::abs(mn),std::abs(mx));
            lo[a]=std::max(0,int(std::floor((mn-margin)/8)));
            hi[a]=std::min(int((s.dimensions[a]-2)/8),int(std::floor((mx+margin)/8)));
        }
        visit(visit,t,lo,hi);
    }
    out.blocks=blocks.size();
    out.candidates_ms=elapsed(start);start=Clock::now();
    std::vector<Interval> cell_rows,sample_rows;
    cell_rows.reserve(blocks.size()*64);sample_rows.reserve(blocks.size()*81);
    for(auto k:blocks) {
        int x=k%nx,y=k/nx%ny,z=k/(std::uint64_t(nx)*ny);
        int ex=std::min(x+8,int(nx)-1),ey=std::min(y+8,int(ny)-1),ez=std::min(z+8,int(s.dimensions[2])-1);
        for(int c=z;c<ez;++c) for(int b=y;b<ey;++b) cell_rows.push_back({key(x,b,c),key(ex-1,b,c)});
        for(int c=z;c<=ez;++c) for(int b=y;b<=ey;++b) sample_rows.push_back({key(x,b,c),key(ex,b,c)});
    }
    out.cells=expand(cell_rows,std::min<std::uint64_t>(INT32_MAX,options.max_memory_bytes/24));
    out.samples=expand(sample_rows,std::min<std::uint64_t>(INT32_MAX/3,options.max_memory_bytes/40));
    out.keys_ms=elapsed(start);start=Clock::now();
    using BuildGrid=nanovdb::tools::build::Grid<std::uint64_t>;
    BuildGrid grid(0);
    auto acc=grid.getAccessor();
    for(std::size_t i=0;i<out.samples.size();++i) {
        auto k=out.samples[i];
        acc.setValue(nanovdb::Coord(int(k%nx),int(k/nx%ny),int(k/(std::uint64_t(nx)*ny))),i+1);
    }
    auto handle=nanovdb::tools::createNanoGrid<BuildGrid,std::uint64_t>(grid,nanovdb::tools::StatsMode::BBox);
    if(handle.bufferSize()>options.max_memory_bytes) throw ResourceError("NanoVDB topology exceeds max_memory_bytes");
    out.grid.resize(handle.bufferSize());
    std::memcpy(out.grid.data(),handle.data(),out.grid.size());
    out.index_ms=elapsed(start);
    return out;
}
}
