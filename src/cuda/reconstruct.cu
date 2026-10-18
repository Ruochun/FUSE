#include "geometry/queries.cuh"
#include "meshing/cell.h"
#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>
#include <chrono>
#include <string>

namespace fuse::detail {
namespace {
using Clock=std::chrono::steady_clock;
double ms(Clock::time_point start) { return std::chrono::duration<double,std::milli>(Clock::now()-start).count(); }
void check(cudaError_t status,const char* what) {
    if(status==cudaErrorMemoryAllocation) throw ResourceError(std::string("CUDA allocation failed: ")+what);
    if(status!=cudaSuccess) throw std::runtime_error(std::string("FUSE CUDA: ")+what+": "+cudaGetErrorString(status));
}
struct Device {
    int old;
    explicit Device(int d) {
        int count=0;check(cudaGetDeviceCount(&count),"enumerate devices (check NVIDIA driver)");
        if(!count) throw std::runtime_error("FUSE CUDA: no visible NVIDIA GPU");
        if(d>=count) throw std::invalid_argument("device is outside the visible GPU range");
        check(cudaGetDevice(&old),"get current device");check(cudaSetDevice(d),"select device");
    }
    ~Device() { cudaSetDevice(old); }
};
struct Budget {
    std::uint64_t limit,live=0,peak=0;
};
struct Buffer {
    void* data=nullptr;
    Budget& budget;
    std::uint64_t size=0;
    Buffer(Budget& b,std::uint64_t bytes):budget(b) {
        if(bytes>budget.limit-budget.live) throw ResourceError("reconstruction exceeds max_memory_bytes; increase voxel_size or the memory budget");
        if(bytes) check(cudaMalloc(&data,bytes),"allocate reconstruction buffer");
        size=bytes;budget.live+=bytes;budget.peak=std::max(budget.peak,budget.live);
    }
    ~Buffer() { if(data) cudaFree(data);budget.live-=size; }
    Buffer(const Buffer&)=delete;
    template<class T> T* as() const { return static_cast<T*>(data); }
};
struct Stream {
    cudaStream_t value=nullptr;
    Stream() { check(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking),"create stream"); }
    ~Stream() { cudaStreamSynchronize(value);cudaStreamDestroy(value); }
    void sync() { check(cudaStreamSynchronize(value),"synchronize reconstruction"); }
};
struct Grid {
    std::uint32_t nx,ny,nz;
    __host__ __device__ std::uint64_t samples() const { return std::uint64_t(nx)*ny*nz; }
    __host__ __device__ std::uint64_t cells() const { return std::uint64_t(nx-1)*(ny-1)*(nz-1); }
    __device__ std::uint64_t index(int x,int y,int z) const { return (std::uint64_t(z)*ny+y)*nx+x; }
    __device__ Vec point(std::uint64_t i) const { return {double(i%nx),double(i/nx%ny),double(i/(std::uint64_t(nx)*ny))}; }
    __device__ Vec cell_point(std::uint64_t i) const { return {double(i%(nx-1)),double(i/(nx-1)%(ny-1)),double(i/(std::uint64_t(nx-1)*(ny-1)))}; }
};
__device__ std::uint64_t thread_index() { return std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x; }
__global__ void evaluate(Grid grid,const Triangle* tris,const Node* nodes,const Shape* shapes,std::uint32_t shape_count,double* field) {
    auto i=thread_index();if(i>=grid.samples()) return;
    Vec p=grid.point(i);double value=INFINITY;
    for(std::uint32_t s=0;s<shape_count;++s) {
        const Shape& shape=shapes[s];
        double lower=shape.bounds.distance2(p);
        if(!shape.bounds.contains(p) && (value<0 || lower>value*value)) continue;
        value=minval(value,signed_distance(p,shape,nodes,tris));
    }
    // Exact/near-zero samples are exterior with a tiny positive symbolic offset.
    // Samples within the numerical zero threshold use the exterior convention.
    field[i]=fabs(value)<=1e-12 ? 1e-7:value;
}
__device__ Cell cell_at(Grid g,const double* field,Vec p) {
    Cell c{};
    for(int i=0;i<8;++i) { Vec q=p+Cell::corner(i); c.values[i]=field[g.index(int(q.x),int(q.y),int(q.z))]; }
    c.build();return c;
}
__global__ void edge_counts(Grid g,const double* field,std::uint64_t* edges) {
    auto key=thread_index();if(key>=3*g.samples()) return;
    auto i=key/3;int axis=key%3;
    Vec p=g.point(i);bool valid=axis==0 ? p.x+1<g.nx : axis==1 ? p.y+1<g.ny : p.z+1<g.nz;
    auto step=axis==0 ? 1ull : axis==1 ? std::uint64_t(g.nx):std::uint64_t(g.nx)*g.ny;
    edges[key]=valid && ((field[i]<0)!=(field[i+step]<0));
}
__global__ void cell_counts(Grid g,const double* field,std::uint64_t* centers,std::uint64_t* faces) {
    auto i=thread_index();if(i>=g.cells()) return;
    Cell c=cell_at(g,field,g.cell_point(i));unsigned seen=0;int edges[12];
    std::uint64_t loops=0,triangles=0;
    for(int e=0;e<12;++e) if(c.degree[e] && !(seen&(1u<<e))) { triangles+=c.loop(e,edges,seen);++loops; }
    centers[i]=loops;faces[i]=triangles;
}
__global__ void emit_edges(Grid g,const double* field,const std::uint64_t* prefix,Vec* vertices) {
    auto key=thread_index();if(key>=3*g.samples()) return;
    auto count=prefix[key];if(count==(key?prefix[key-1]:0)) return;
    auto i=key/3;int axis=key%3;
    auto step=axis==0 ? 1ull : axis==1 ? std::uint64_t(g.nx):std::uint64_t(g.nx)*g.ny;
    double t=field[i]/(field[i]-field[i+step]);Vec p=g.point(i);
    if(axis==0) p.x+=t;else if(axis==1) p.y+=t;else p.z+=t;
    vertices[count-1]=p;
}
__device__ std::uint64_t edge_id(Grid g,Vec cell,int e,const std::uint64_t* edges) {
    int a,b;Cell::endpoints(e,a,b);Vec p=Cell::corner(a),q=Cell::corner(b);
    int axis=p.x!=q.x ? 0:p.y!=q.y ? 1:2;
    Vec lo{minval(p.x,q.x)+cell.x,minval(p.y,q.y)+cell.y,minval(p.z,q.z)+cell.z};
    auto key=3*g.index(int(lo.x),int(lo.y),int(lo.z))+axis;
    return edges[key]-1;
}
__global__ void emit_cells(Grid g,const double* field,const std::uint64_t* edge_prefix,
                           const std::uint64_t* center_prefix,const std::uint64_t* face_prefix,
                           std::uint64_t edge_count,Vec* vertices,std::uint64_t* faces) {
    auto i=thread_index();if(i>=g.cells()) return;
    Vec p=g.cell_point(i);Cell c=cell_at(g,field,p);unsigned seen=0;
    auto center=edge_count+(i?center_prefix[i-1]:0),face=i?face_prefix[i-1]:0;
    for(int e=0;e<12;++e) if(c.degree[e] && !(seen&(1u<<e))) {
        Vec mean{0,0,0};int n=0,previous=-1,current=e;
        do {
            mean=mean+c.position(current);++n;seen|=1u<<current;
            int next=c.adjacent[current][0]==previous ? c.adjacent[current][1]:c.adjacent[current][0];
            previous=current;current=next;
        } while(current!=e);
        vertices[center]=p+mean*(1.0/n);
        bool reverse=c.reverse_loop(e,c.adjacent[e][0]);
        previous=-1;current=e;
        do {
            int next=c.adjacent[current][0]==previous ? c.adjacent[current][1]:c.adjacent[current][0];
            auto a=edge_id(g,p,current,edge_prefix),b=edge_id(g,p,next,edge_prefix);
            faces[3*face]=reverse?b:a;faces[3*face+1]=reverse?a:b;faces[3*face+2]=center;++face;
            previous=current;current=next;
        } while(current!=e);
        ++center;
    }
}
unsigned blocks(std::uint64_t n) { return static_cast<unsigned>((n+255)/256); }
void launch_check() { check(cudaGetLastError(),"launch reconstruction kernel"); }
} // namespace
ReconstructionResult run_cuda(const Scene& scene,const ReconstructionOptions& options,
                              const std::vector<double>* sampled_field) {
    Device device(options.device);
    Budget budget{options.max_memory_bytes};
    Grid grid{scene.dimensions[0],scene.dimensions[1],scene.dimensions[2]};
    ReconstructionResult out;
    auto start=Clock::now();
    Buffer tris(budget,scene.triangles.size()*sizeof(Triangle)),nodes(budget,scene.nodes.size()*sizeof(Node)),
           shapes(budget,scene.shapes.size()*sizeof(Shape)),field(budget,grid.samples()*sizeof(double));
    // All host copies stay alive until the stream has synchronized.
    Stream stream;
    check(cudaMemcpyAsync(tris.data,scene.triangles.data(),tris.size,cudaMemcpyHostToDevice,stream.value),"upload triangles");
    check(cudaMemcpyAsync(nodes.data,scene.nodes.data(),nodes.size,cudaMemcpyHostToDevice,stream.value),"upload BVH");
    check(cudaMemcpyAsync(shapes.data,scene.shapes.data(),shapes.size,cudaMemcpyHostToDevice,stream.value),"upload shapes");
    stream.sync();out.upload_ms=ms(start);start=Clock::now();
    if(sampled_field) {
        if(sampled_field->size()!=grid.samples()) throw std::invalid_argument("sampled field shape mismatch");
        for(double v:*sampled_field) if(!std::isfinite(v) || v==0)
            throw std::invalid_argument("internal extraction test requires finite nonzero samples");
        check(cudaMemcpyAsync(field.data,sampled_field->data(),field.size,cudaMemcpyHostToDevice,stream.value),"upload validation field");
    } else {
        evaluate<<<blocks(grid.samples()),256,0,stream.value>>>(grid,tris.as<Triangle>(),nodes.as<Node>(),shapes.as<Shape>(),scene.shapes.size(),field.as<double>());
        launch_check();
    }
    stream.sync();out.field_ms=ms(start);start=Clock::now();
    Buffer edges(budget,3*grid.samples()*sizeof(std::uint64_t)),centers(budget,grid.cells()*sizeof(std::uint64_t)),
           faces(budget,grid.cells()*sizeof(std::uint64_t));
    std::size_t scratch_bytes=0;
    for(Buffer* b:{&edges,&centers}) {
        std::size_t required=0;
        check(cub::DeviceScan::InclusiveSum(nullptr,required,b->as<std::uint64_t>(),b->as<std::uint64_t>(),int(b->size/8),stream.value),"size scan scratch");
        scratch_bytes=std::max(scratch_bytes,required);
    }
    Buffer scratch(budget,scratch_bytes);
    edge_counts<<<blocks(3*grid.samples()),256,0,stream.value>>>(grid,field.as<double>(),edges.as<std::uint64_t>());
    launch_check();
    cell_counts<<<blocks(grid.cells()),256,0,stream.value>>>(grid,field.as<double>(),centers.as<std::uint64_t>(),faces.as<std::uint64_t>());
    launch_check();
    for(Buffer* b:{&edges,&centers,&faces}) {
        std::size_t available=scratch.size;
        check(cub::DeviceScan::InclusiveSum(scratch.data,available,b->as<std::uint64_t>(),b->as<std::uint64_t>(),int(b->size/8),stream.value),"scan output counts");
    }
    std::uint64_t ne=0,nc=0,nf=0;
    check(cudaMemcpyAsync(&ne,edges.as<std::uint64_t>()+3*grid.samples()-1,8,cudaMemcpyDeviceToHost,stream.value),"read edge count");
    check(cudaMemcpyAsync(&nc,centers.as<std::uint64_t>()+grid.cells()-1,8,cudaMemcpyDeviceToHost,stream.value),"read center count");
    check(cudaMemcpyAsync(&nf,faces.as<std::uint64_t>()+grid.cells()-1,8,cudaMemcpyDeviceToHost,stream.value),"read triangle count");
    stream.sync();
    if(ne+nc>SIZE_MAX/(3*sizeof(double)) || nf>SIZE_MAX/(3*sizeof(std::uint64_t)))
        throw ResourceError("output index or allocation overflow");
    Buffer vertices(budget,(ne+nc)*sizeof(Vec)),triangles(budget,nf*3*sizeof(std::uint64_t));
    if(ne) {
        emit_edges<<<blocks(3*grid.samples()),256,0,stream.value>>>(grid,field.as<double>(),edges.as<std::uint64_t>(),vertices.as<Vec>());
        launch_check();
        emit_cells<<<blocks(grid.cells()),256,0,stream.value>>>(grid,field.as<double>(),edges.as<std::uint64_t>(),centers.as<std::uint64_t>(),faces.as<std::uint64_t>(),ne,vertices.as<Vec>(),triangles.as<std::uint64_t>());
        launch_check();
    }
    stream.sync();out.extraction_ms=ms(start);start=Clock::now();
    out.vertices.resize((ne+nc)*3);out.faces.resize(nf*3);
    if(vertices.size) check(cudaMemcpy(out.vertices.data(),vertices.data,vertices.size,cudaMemcpyDeviceToHost),"download vertices");
    if(triangles.size) check(cudaMemcpy(out.faces.data(),triangles.data,triangles.size,cudaMemcpyDeviceToHost),"download faces");
    if(options.return_field) {
        out.field.resize(grid.samples());
        check(cudaMemcpy(out.field.data(),field.data,field.size,cudaMemcpyDeviceToHost),"download diagnostic field");
        for(auto& f:out.field) f*=options.voxel_size;
    }
    for(std::size_t i=0;i<out.vertices.size();++i)
        out.vertices[i]=scene.origin[i%3]+out.vertices[i]*options.voxel_size;
    // Do not silently drop triangles if world-coordinate rounding collapses them.
    for(std::size_t i=0;i<out.faces.size();i+=3) {
        Vec p[3];
        for(int k=0;k<3;++k) {
            if(out.faces[i+k]>=out.vertices.size()/3) throw std::runtime_error("internal extraction error: invalid vertex index");
            auto v=out.faces[i+k]*3;
            p[k]={out.vertices[v],out.vertices[v+1],out.vertices[v+2]};
        }
        // Check area in grid units: valid world-unit scales can otherwise make
        // the squared area overflow/underflow while every coordinate is finite.
        auto n=cross((p[1]-p[0])*(1.0/options.voxel_size),
                     (p[2]-p[0])*(1.0/options.voxel_size));
        if(!(dot(n,n)>0) || !std::isfinite(dot(n,n)))
            throw std::runtime_error("output triangle "+std::to_string(i/3)+
                " collapsed at this resolution/coordinate precision; change voxel_size or translate the scene");
    }
    out.download_ms=ms(start);out.peak_device_bytes=budget.peak;
    return out;
}
} // namespace fuse::detail
