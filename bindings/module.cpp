#include <fuse/smoke.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <memory>
#include <vector>

namespace nb = nanobind;

NB_MODULE(_native, m) {
    m.doc() = "FUSE native build proof (M0)";
    m.def("smoke_test", [](int device) {
        std::unique_ptr<std::vector<float>> values;
        {
            nb::gil_scoped_release release;
            values = std::make_unique<std::vector<float>>(fuse::smoke_test(device));
        }
        auto* storage = values.get();
        nb::capsule owner(storage, [](void* p) noexcept {
            delete static_cast<std::vector<float>*>(p);
        });
        values.release();
        return nb::ndarray<nb::numpy, float>(storage->data(), {storage->size()}, owner);
    }, nb::arg("device") = 0,
    "Read a fixed NanoVDB grid on CUDA; returns float32 [3, -2, 0.5, 7].");
    m.def("build_info", [] {
        nb::dict info;
        info["nanovdb_revision"] = fuse::nanovdb_revision();
        info["cuda_version"] = fuse::cuda_build_version();
        info["cuda_architectures"] = fuse::cuda_architectures();
        info["cuda_runtime"] = "static";
        info["milestone"] = 0;
        return info;
    });
}
