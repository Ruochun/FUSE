#include <fuse/smoke.h>
#include <fuse/build_config.h>
#include <cuda_runtime.h>
#include <nanovdb/tools/GridBuilder.h>
#include <nanovdb/tools/CreateNanoGrid.h>

#include <stdexcept>
#include <string>

namespace fuse {
namespace {
void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("FUSE CUDA: ") + operation + ": " +
                                 cudaGetErrorString(status));
    }
}
struct DeviceGuard {
    int previous;
    explicit DeviceGuard(int device) {
        int count = 0;
        if (device < 0) throw std::invalid_argument("device must be nonnegative");
        check(cudaGetDeviceCount(&count), "enumerate devices (check NVIDIA driver)");
        if (count == 0) throw std::runtime_error("FUSE CUDA: no visible NVIDIA GPU");
        if (device >= count) throw std::invalid_argument("device is outside the visible GPU range");
        check(cudaGetDevice(&previous), "get current device");
        check(cudaSetDevice(device), "select device");
    }
    ~DeviceGuard() { cudaSetDevice(previous); }
};
struct Stream {
    cudaStream_t value = nullptr;
    Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking), "create stream"); }
    ~Stream() { cudaStreamSynchronize(value); cudaStreamDestroy(value); }
};
struct DeviceMemory {
    void* value = nullptr;
    explicit DeviceMemory(size_t bytes) { check(cudaMalloc(&value, bytes), "allocate smoke buffer"); }
    ~DeviceMemory() { cudaFree(value); }
};
__global__ void read_grid(const nanovdb::NanoGrid<float>* grid, float* output) {
    const int i = threadIdx.x;
    if (i >= 4) return;
    const nanovdb::Coord coords[] = {
        nanovdb::Coord(0, 0, 0), nanovdb::Coord(-9, 2, 3),
        nanovdb::Coord(8, 0, 0), nanovdb::Coord(100, 100, 100)};
    auto accessor = grid->getAccessor();
    output[i] = accessor.getValue(coords[i]);
}
} // namespace

std::vector<float> smoke_test(int device) {
    DeviceGuard guard(device);
    using Grid = nanovdb::tools::build::Grid<float>;
    Grid grid(7.0f);
    auto accessor = grid.getAccessor();
    accessor.setValue(nanovdb::Coord(0, 0, 0), 3.0f);
    accessor.setValue(nanovdb::Coord(-9, 2, 3), -2.0f);
    accessor.setValue(nanovdb::Coord(8, 0, 0), 0.5f);
    auto handle = nanovdb::tools::createNanoGrid<Grid, float>(grid);
    if (!handle.grid<float>()) throw std::runtime_error("FUSE: NanoVDB grid construction failed");
    DeviceMemory device_grid(handle.bufferSize());
    DeviceMemory device_output(4 * sizeof(float));
    // Destroy/synchronize the stream before freeing device or host storage on errors.
    std::vector<float> result(4);
    Stream stream;
    check(cudaMemcpyAsync(device_grid.value, handle.data(), handle.bufferSize(),
                          cudaMemcpyHostToDevice, stream.value), "upload NanoVDB grid");
    read_grid<<<1, 32, 0, stream.value>>>(
        static_cast<const nanovdb::NanoGrid<float>*>(device_grid.value),
        static_cast<float*>(device_output.value));
    check(cudaGetLastError(), "launch NanoVDB read kernel");
    check(cudaMemcpyAsync(result.data(), device_output.value, 4 * sizeof(float),
                          cudaMemcpyDeviceToHost, stream.value), "download samples");
    check(cudaStreamSynchronize(stream.value), "synchronize NanoVDB read kernel");
    return result;
}
const char* nanovdb_revision() noexcept { return FUSE_NANOVDB_REVISION; }
const char* cuda_architectures() noexcept { return FUSE_CUDA_ARCHITECTURES; }
int cuda_build_version() noexcept { return CUDART_VERSION; }
} // namespace fuse
