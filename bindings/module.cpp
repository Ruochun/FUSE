#include <fuse/smoke.h>
#include <fuse/reconstruct.h>
#include <nanobind/stl/array.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <memory>
#include <vector>

namespace nb = nanobind;

NB_MODULE(_native, m) {
    m.doc() = "FUSE sparse CUDA reconstruction backend";
    nb::exception<fuse::ResourceError>(m, "ResourceError", PyExc_MemoryError);
    m.def("_reconstruct", [](nb::list meshes, double voxel_size, int device,
                             std::uint64_t memory, bool topology, bool field, bool sparse) {
        using Vertices = nb::ndarray<nb::numpy, const double, nb::shape<-1,3>, nb::c_contig, nb::device::cpu>;
        using Faces = nb::ndarray<nb::numpy, const std::uint64_t, nb::shape<-1,3>, nb::c_contig, nb::device::cpu>;
        std::vector<fuse::MeshView> views;
        for (auto entry : meshes) {
            auto pair = nb::cast<nb::tuple>(entry);
            if (nb::len(pair) != 2) throw std::invalid_argument("expected a (vertices, faces) pair");
            // Never create short-lived implicit conversion buffers behind MeshView.
            auto v = nb::cast<Vertices>(pair[0], false);
            auto f = nb::cast<Faces>(pair[1], false);
            views.push_back({v.data(), v.shape(0), f.data(), f.shape(0)});
        }
        fuse::ReconstructionOptions options{voxel_size,device,memory,topology,field,sparse};
        std::unique_ptr<fuse::ReconstructionResult> result;
        {
            nb::gil_scoped_release release;
            result = std::make_unique<fuse::ReconstructionResult>(fuse::reconstruct(views,options));
        }
        auto* storage = result.get();
        nb::capsule owner(storage, [](void* p) noexcept { delete static_cast<fuse::ReconstructionResult*>(p); });
        result.release();
        static double empty_value = 0;
        static std::uint64_t empty_index = 0;
        nb::dict output, info, timings;
        output["vertices"] = nb::ndarray<nb::numpy,double>(
            storage->vertices.empty()? &empty_value : storage->vertices.data(), {storage->vertices.size()/3,3}, owner);
        output["faces"] = nb::ndarray<nb::numpy,std::uint64_t>(
            storage->faces.empty()? &empty_index : storage->faces.data(), {storage->faces.size()/3,3}, owner);
        if(field) {
            output["field"] = nb::ndarray<nb::numpy,double>(
                storage->field.empty()? &empty_value : storage->field.data(),
                {storage->grid_shape[2],storage->grid_shape[1],storage->grid_shape[0]},owner);
        } else output["field"] = nb::none();
        info["origin"] = nb::cast(storage->origin);
        info["grid_shape"] = nb::cast(storage->grid_shape);
        info["voxel_size"] = storage->voxel_size;
        info["device"] = storage->device;
        info["peak_device_bytes"] = storage->peak_device_bytes;
        info["vertex_count"] = storage->vertices.size()/3;
        info["triangle_count"] = storage->faces.size()/3;
        info["backend"] = sparse ? "sparse_cuda" : "dense_cuda";
        info["candidate_blocks"] = storage->candidate_blocks;
        info["active_samples"] = storage->active_samples;
        info["candidate_cells"] = storage->candidate_cells;
        info["topology_bytes"] = storage->topology_bytes;
        timings["topology"] = storage->topology_ms;
        info["boundary_mode"] = "full";
        timings["prepare"] = storage->prepare_ms;
        timings["upload"] = storage->upload_ms;
        timings["field"] = storage->field_ms;
        timings["extraction"] = storage->extraction_ms;
        timings["download"] = storage->download_ms;
        info["timings_ms"] = timings;
        output["metadata"] = info;
        return output;
    }, nb::arg("meshes"), nb::arg("voxel_size"), nb::arg("device"),
       nb::arg("max_memory_bytes"), nb::arg("validate_topology"), nb::arg("return_field"), nb::arg("sparse"));
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
        info["milestone"] = 2;
        return info;
    });
}
