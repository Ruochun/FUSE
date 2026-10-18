#include "geometry/queries.cuh"
#include "meshing/cell.h"
#include "volume/topology.h"
#include <nanovdb/NanoVDB.h>
#include <type_traits>
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
        if(bytes>budget.limit-budget.live) throw ResourceError("reconstruction exceeds max_memory_bytes: requested "+std::to_string(bytes)+
            " bytes with "+std::to_string(budget.live)+" live and limit "+std::to_string(budget.limit)+
            "; increase voxel_size or the memory budget");
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
    __device__ Grid getAccessor() const { return *this; }
};
struct SparseGrid : Grid {
    const nanovdb::NanoGrid<std::uint64_t>* topology = nullptr;
    const std::uint64_t* sample_keys = nullptr;
    const std::uint64_t* cell_keys = nullptr;
    std::uint64_t sample_count = 0, cell_count = 0;
    __host__ __device__ std::uint64_t samples() const { return sample_count; }
    __host__ __device__ std::uint64_t cells() const { return cell_count; }
    __device__ std::uint64_t index(int x,int y,int z) const {
        return topology->tree().getValue(nanovdb::Coord(x,y,z))-1;
    }
    struct Accessor {
        nanovdb::DefaultReadAccessor<std::uint64_t> values;
        __device__ std::uint64_t index(int x,int y,int z) { return values.getValue(nanovdb::Coord(x,y,z))-1; }
    };
    __device__ Accessor getAccessor() const { return {topology->getAccessor()}; }
    __device__ Vec point(std::uint64_t i) const { return Grid::point(sample_keys[i]); }
    __device__ Vec cell_point(std::uint64_t i) const { return Grid::point(cell_keys[i]); }
};
__device__ std::uint64_t thread_index() { return std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x; }
__device__ void evaluate_shape(Vec p,const Shape& shape,const Node* nodes,const Triangle* tris,double& value) {
    double lower=shape.bounds.distance2(p);
    if(!shape.bounds.contains(p) && (value<0 || lower>value*value)) return;
    value=minval(value,signed_distance(p,shape,nodes,tris));
}
template<class G,bool Hierarchy> __global__ void evaluate(G grid,const Triangle* tris,const Node* nodes,const Shape* shapes,std::uint32_t shape_count,const Node* shape_nodes,std::uint32_t shape_node_count,double* field, std::uint64_t offset=0, std::uint64_t count=UINT64_MAX) {
    auto local=thread_index();auto i=local+offset;if(local>=count || i>=grid.samples()) return;
    Vec p=grid.point(i);double value=INFINITY;
    if constexpr(Hierarchy) {
        // As with the triangle BVH, median splitting bounds the depth below 32.
        // Visit nearby shape bounds first to establish a useful pruning value.
        std::uint32_t pending[32],node=0;int count=0;
        while(true) {
            const auto& n=shape_nodes[node];
            const double lower=n.bounds.distance2(p);
            // A possible containing solid must always contribute, regardless of
            // distance to its surface or the sign of an already evaluated shape.
            if(n.bounds.contains(p) || !(value<0 || lower>value*value)) {
                if(n.count) {
                    for(std::uint32_t j=0;j<n.count;++j) evaluate_shape(p,shapes[n.begin+j],nodes,tris,value);
                } else {
                    auto near=node+1,far=shape_nodes[near].escape;
                    if(shape_nodes[far].bounds.distance2(p)<shape_nodes[near].bounds.distance2(p)) {
                        auto swap=near;near=far;far=swap;
                    }
                    pending[count++]=far;node=near;continue;
                }
            }
            if(!count) break;
            node=pending[--count];
        }
    } else {
        for(std::uint32_t s=0;s<shape_count;++s) evaluate_shape(p,shapes[s],nodes,tris,value);
    }
    // Exact/near-zero samples are exterior with a tiny positive symbolic offset.
    // Samples within the numerical zero threshold use the exterior convention.
    field[local]=fabs(value)<=1e-12 ? 1e-7:value;
}
template<class G> __device__ Cell cell_at(G& g,const double* field,Vec p) {
    Cell c{};
    for(int i=0;i<8;++i) { Vec q=p+Cell::corner(i); c.values[i]=field[g.index(int(q.x),int(q.y),int(q.z))]; }
    c.build();return c;
}
template<class G> __global__ void edge_counts(G g,const double* field,std::uint64_t* edges) {
    auto key=thread_index();if(key>=3*g.samples()) return;
    auto i=key/3;int axis=key%3;
    Vec p=g.point(i);bool valid=axis==0 ? p.x+1<g.nx : axis==1 ? p.y+1<g.ny : p.z+1<g.nz;
    auto next=valid ? g.index(int(p.x)+(axis==0),int(p.y)+(axis==1),int(p.z)+(axis==2)) : UINT64_MAX;
    edges[key]=next!=UINT64_MAX && ((field[i]<0)!=(field[next]<0));
}
template<class G> __global__ void cell_counts(G g,const double* field,std::uint64_t* centers,std::uint64_t* faces) {
    auto i=thread_index();if(i>=g.cells()) return;
    auto accessor=g.getAccessor();
    Cell c=cell_at(accessor,field,g.cell_point(i));unsigned seen=0;int edges[12];
    std::uint64_t loops=0,triangles=0;
    for(int e=0;e<12;++e) if(c.degree[e] && !(seen&(1u<<e))) { triangles+=c.loop(e,edges,seen);++loops; }
    centers[i]=loops;faces[i]=triangles;
}
template<class G> __global__ void emit_edges(G g,const double* field,const std::uint64_t* prefix,Vec* vertices) {
    auto key=thread_index();if(key>=3*g.samples()) return;
    auto count=prefix[key];if(count==(key?prefix[key-1]:0)) return;
    auto i=key/3;int axis=key%3;
    Vec p=g.point(i);
    auto next=g.index(int(p.x)+(axis==0),int(p.y)+(axis==1),int(p.z)+(axis==2));
    double t=field[i]/(field[i]-field[next]);
    if(axis==0) p.x+=t;else if(axis==1) p.y+=t;else p.z+=t;
    vertices[count-1]=p;
}
template<class G> __device__ std::uint64_t edge_id(G& g,Vec cell,int e,const std::uint64_t* edges) {
    int a,b;Cell::endpoints(e,a,b);Vec p=Cell::corner(a),q=Cell::corner(b);
    int axis=p.x!=q.x ? 0:p.y!=q.y ? 1:2;
    Vec lo{minval(p.x,q.x)+cell.x,minval(p.y,q.y)+cell.y,minval(p.z,q.z)+cell.z};
    auto key=3*g.index(int(lo.x),int(lo.y),int(lo.z))+axis;
    return edges[key]-1;
}
template<class G> __global__ void emit_cells(G g,const double* field,const std::uint64_t* edge_prefix,
                           const std::uint64_t* center_prefix,const std::uint64_t* face_prefix,
                           std::uint64_t edge_count,Vec* vertices,std::uint64_t* faces) {
    auto i=thread_index();if(i>=g.cells()) return;
    auto accessor=g.getAccessor();
    Vec p=g.cell_point(i);Cell c=cell_at(accessor,field,p);unsigned seen=0;
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
            auto a=edge_id(accessor,p,current,edge_prefix),b=edge_id(accessor,p,next,edge_prefix);
            faces[3*face]=reverse?b:a;faces[3*face+1]=reverse?a:b;faces[3*face+2]=center;++face;
            previous=current;current=next;
        } while(current!=e);
        ++center;
    }
}
unsigned blocks(std::uint64_t n) { return static_cast<unsigned>((n+255)/256); }
template<class G> void launch_evaluate(G grid,const Triangle* tris,const Node* nodes,const Shape* shapes,
    std::uint32_t shape_count,const Node* shape_nodes,std::uint32_t shape_node_count,double* field,
    cudaStream_t stream,std::uint64_t offset=0,std::uint64_t count=UINT64_MAX) {
    const auto n=std::min(count,grid.samples()-offset);
    // Separate kernels avoid imposing hierarchy traversal state on small scenes.
    if(shape_node_count)
        evaluate<G,true><<<blocks(n),256,0,stream>>>(grid,tris,nodes,shapes,shape_count,shape_nodes,shape_node_count,field,offset,count);
    else
        evaluate<G,false><<<blocks(n),256,0,stream>>>(grid,tris,nodes,shapes,shape_count,shape_nodes,shape_node_count,field,offset,count);
}
void launch_check() { check(cudaGetLastError(),"launch reconstruction kernel"); }
} // namespace
template<class G> ReconstructionResult run_grid(const Scene& scene,const ReconstructionOptions& options,
                              G grid, const SparseTopology* topology, const std::vector<double>* sampled_field) {
    Device device(options.device);
    Budget budget{options.max_memory_bytes};
    ReconstructionResult out;
    out.active_samples=grid.samples();out.candidate_cells=grid.cells();
    if(topology) { out.candidate_blocks=topology->blocks;out.topology_bytes=topology->grid.size(); }
    auto start=Clock::now();
    Buffer tris(budget,scene.triangles.size()*sizeof(Triangle)),nodes(budget,scene.nodes.size()*sizeof(Node)),
           shapes(budget,scene.shapes.size()*sizeof(Shape)),shape_nodes(budget,scene.shape_nodes.size()*sizeof(Node)),field(budget,grid.samples()*sizeof(double));
    Buffer nano(budget,topology?topology->grid.size():0),
           sample_keys(budget,topology?topology->samples.size()*8:0),
           cell_keys(budget,topology?topology->cells.size()*8:0);
    // All host copies stay alive until the stream has synchronized.
    Stream stream;
    if constexpr(std::is_same_v<G,SparseGrid>) {
        check(cudaMemcpyAsync(nano.data,topology->grid.data(),nano.size,cudaMemcpyHostToDevice,stream.value),"upload NanoVDB topology");
        check(cudaMemcpyAsync(sample_keys.data,topology->samples.data(),sample_keys.size,cudaMemcpyHostToDevice,stream.value),"upload sample keys");
        check(cudaMemcpyAsync(cell_keys.data,topology->cells.data(),cell_keys.size,cudaMemcpyHostToDevice,stream.value),"upload cell keys");
        grid.topology=nano.as<nanovdb::NanoGrid<std::uint64_t>>();
        grid.sample_keys=sample_keys.as<std::uint64_t>();grid.cell_keys=cell_keys.as<std::uint64_t>();
    }
    check(cudaMemcpyAsync(tris.data,scene.triangles.data(),tris.size,cudaMemcpyHostToDevice,stream.value),"upload triangles");
    check(cudaMemcpyAsync(nodes.data,scene.nodes.data(),nodes.size,cudaMemcpyHostToDevice,stream.value),"upload BVH");
    check(cudaMemcpyAsync(shapes.data,scene.shapes.data(),shapes.size,cudaMemcpyHostToDevice,stream.value),"upload shapes");
    if(shape_nodes.size) check(cudaMemcpyAsync(shape_nodes.data,scene.shape_nodes.data(),shape_nodes.size,cudaMemcpyHostToDevice,stream.value),"upload shape hierarchy");
    stream.sync();out.upload_ms=ms(start);start=Clock::now();
    if(sampled_field) {
        if(sampled_field->size()!=Grid{grid.nx,grid.ny,grid.nz}.samples()) throw std::invalid_argument("sampled field shape mismatch");
        for(double v:*sampled_field) if(!std::isfinite(v) || v==0)
            throw std::invalid_argument("internal extraction test requires finite nonzero samples");
        if constexpr(std::is_same_v<G,SparseGrid>) {
            std::vector<double> active(grid.samples());
            for(std::size_t i=0;i<active.size();++i) active[i]=(*sampled_field)[topology->samples[i]];
            check(cudaMemcpyAsync(field.data,active.data(),field.size,cudaMemcpyHostToDevice,stream.value),"upload sparse validation field");
            stream.sync(); // active owns the source storage
        } else {
            check(cudaMemcpyAsync(field.data,sampled_field->data(),field.size,cudaMemcpyHostToDevice,stream.value),"upload validation field");
        }
    } else {
        launch_evaluate(grid,tris.as<Triangle>(),nodes.as<Node>(),shapes.as<Shape>(),scene.shapes.size(),shape_nodes.as<Node>(),scene.shape_nodes.size(),field.as<double>(),stream.value);
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
        if constexpr(std::is_same_v<G,SparseGrid>) {
            Grid dense{grid.nx,grid.ny,grid.nz};
            out.field.resize(dense.samples());
            // Diagnostics are explicitly dense, but never require a dense GPU
            // allocation. Missing topology is queried, never filled as exterior.
            auto chunk=std::min<std::uint64_t>(65536,(budget.limit-budget.live)/8);
            if(!chunk) throw ResourceError("diagnostic scratch exceeds max_memory_bytes");
            Buffer diagnostic(budget,chunk*8);
            for(std::uint64_t offset=0;offset<dense.samples();offset+=chunk) {
                auto n=std::min(chunk,dense.samples()-offset);
                launch_evaluate(dense,tris.as<Triangle>(),nodes.as<Node>(),shapes.as<Shape>(),scene.shapes.size(),shape_nodes.as<Node>(),scene.shape_nodes.size(),diagnostic.as<double>(),stream.value,offset,n);
                launch_check();
                check(cudaMemcpyAsync(out.field.data()+offset,diagnostic.data,n*8,cudaMemcpyDeviceToHost,stream.value),"download diagnostic chunk");
                stream.sync();
            }
            // Report the actual values consumed by sparse extraction at its
            // active samples, so dense/sparse field comparisons test that path.
            std::vector<double> active(grid.samples());
            check(cudaMemcpy(active.data(),field.data,field.size,cudaMemcpyDeviceToHost),"download sparse field");
            for(std::size_t i=0;i<active.size();++i) out.field[topology->samples[i]]=active[i];
        } else {
            out.field.resize(grid.samples());
            check(cudaMemcpy(out.field.data(),field.data,field.size,cudaMemcpyDeviceToHost),"download diagnostic field");
        }
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
ReconstructionResult run_cuda(const Scene& scene,const ReconstructionOptions& options,
                              const std::vector<double>* sampled_field) {
    Grid dense{scene.dimensions[0],scene.dimensions[1],scene.dimensions[2]};
    if(options.return_field && dense.samples()>options.max_memory_bytes/sizeof(double))
        throw ResourceError("dense diagnostic field exceeds max_memory_bytes; disable return_field or increase the budget");
    if(!options.sparse) return run_grid(scene,options,dense,nullptr,sampled_field);
    // Validate CUDA availability before potentially expensive host topology work.
    Device device(options.device);
    auto start=Clock::now();
    auto topology=make_topology(scene,options);
    double elapsed=ms(start);
    SparseGrid sparse; sparse.nx=dense.nx;sparse.ny=dense.ny;sparse.nz=dense.nz;
    sparse.sample_count=topology.samples.size();sparse.cell_count=topology.cells.size();
    auto result=run_grid(scene,options,sparse,&topology,sampled_field);
    result.topology_ms=elapsed;
    result.topology_candidates_ms=topology.candidates_ms;
    result.topology_keys_ms=topology.keys_ms;
    result.topology_index_ms=topology.index_ms;
    return result;
}
} // namespace fuse::detail
