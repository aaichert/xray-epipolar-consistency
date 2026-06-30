#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "LibRayCastBackproject/VoxelData.h"
#include "LibRayCastBackproject/VolumeRendering.h"
#include "LibUtilsCuda/CudaBindlessTexture.h"
#include "HeaderOnly/NRRD/nrrd.hxx"

namespace py = pybind11;

// Declare the external CUDA E/A raycast function
extern void raycast_ea_1Dtf(
    int n_u, int n_v, int n_c,
    float * pixel_data_d,
    float * model_C_Pinv_d,
    cudaTextureObject_t voxel_data,
    float * ray_entry_d,
    float * ray_exit_d,
    float * noise_d,
    float samples_per_voxel,
    cudaTextureObject_t tf
);

// Declare the external CUDA E/A shaded raycast function
extern void raycast_ea_shaded_1Dtf(
    int n_u, int n_v, int n_c,
    float * pixel_data_d,
    float * model_C_Pinv_d,
    cudaTextureObject_t voxel_data,
    float * ray_entry_d,
    float * ray_exit_d,
    float * noise_d,
    float samples_per_voxel,
    cudaTextureObject_t tf
);

// Define EmissionAbsorption pass in C++
class EmissionAbsorption : public VolumeRendering::RaycastPass {
    std::unique_ptr<UtilsCuda::BindlessTexture1D<float4>> tf_texture;
    float min_val = std::numeric_limits<float>::quiet_NaN();
    float max_val = std::numeric_limits<float>::quiet_NaN();
public:
    EmissionAbsorption() {}

    void setTransferFunction(py::array_t<float, py::array::c_style | py::array::forcecast> tf_data) {
        if (tf_data.ndim() != 2 || tf_data.shape(1) != 4) {
            throw std::runtime_error("Transfer function must be a 2D array of shape (N, 4) containing RGBA values.");
        }
        int w = tf_data.shape(0);
        // Create BindlessTexture1D<float4>
        tf_texture = std::make_unique<UtilsCuda::BindlessTexture1D<float4>>(w, (const float4*)tf_data.data());
    }

    void setRange(float min_v, float max_v) {
        min_val = min_v;
        max_val = max_v;
    }

    virtual float getMinValue() override { return min_val; }
    virtual float getMaxValue() override { return max_val; }

    virtual void render(
        int n_u, int n_v, int n_c,
        float * pixel_data_d,
        float * model_C_Pinv_d,
        const UtilsCuda::BindlessTexture3D<float>& voxel_data,
        float * ray_entry_d,
        float * ray_exit_d,
        float * noise_d,
        float samples_per_voxel
    ) override {
        if (!tf_texture) {
            throw std::runtime_error("Transfer function has not been set for the Emission-Absorption pass.");
        }
        raycast_ea_1Dtf(
            n_u, n_v, n_c,
            pixel_data_d,
            model_C_Pinv_d,
            voxel_data,
            ray_entry_d,
            ray_exit_d,
            noise_d,
            samples_per_voxel,
            *tf_texture
        );
    }
};

// Define EmissionAbsorptionShaded pass in C++
class EmissionAbsorptionShaded : public VolumeRendering::RaycastPass {
    std::unique_ptr<UtilsCuda::BindlessTexture1D<float4>> tf_texture;
    float min_val = std::numeric_limits<float>::quiet_NaN();
    float max_val = std::numeric_limits<float>::quiet_NaN();
public:
    EmissionAbsorptionShaded() {}

    void setTransferFunction(py::array_t<float, py::array::c_style | py::array::forcecast> tf_data) {
        if (tf_data.ndim() != 2 || tf_data.shape(1) != 4) {
            throw std::runtime_error("Transfer function must be a 2D array of shape (N, 4) containing RGBA values.");
        }
        int w = tf_data.shape(0);
        tf_texture = std::make_unique<UtilsCuda::BindlessTexture1D<float4>>(w, (const float4*)tf_data.data());
    }

    void setRange(float min_v, float max_v) {
        min_val = min_v;
        max_val = max_v;
    }

    virtual float getMinValue() override { return min_val; }
    virtual float getMaxValue() override { return max_val; }

    virtual void render(
        int n_u, int n_v, int n_c,
        float * pixel_data_d,
        float * model_C_Pinv_d,
        const UtilsCuda::BindlessTexture3D<float>& voxel_data,
        float * ray_entry_d,
        float * ray_exit_d,
        float * noise_d,
        float samples_per_voxel
    ) override {
        if (!tf_texture) {
            throw std::runtime_error("Transfer function has not been set for the Emission-Absorption-Shaded pass.");
        }
        raycast_ea_shaded_1Dtf(
            n_u, n_v, n_c,
            pixel_data_d,
            model_C_Pinv_d,
            voxel_data,
            ray_entry_d,
            ray_exit_d,
            noise_d,
            samples_per_voxel,
            *tf_texture
        );
    }
};

class PyVoxelData {
    std::shared_ptr<VolumeRendering::VoxelData> vd;
public:
    PyVoxelData(py::array_t<float, py::array::c_style | py::array::forcecast> volume, bool use_ess=true) {
        if (volume.ndim() != 3) {
            throw std::runtime_error("Volume array must be 3D");
        }
        int d = volume.shape(0);
        int h = volume.shape(1);
        int w = volume.shape(2);

        NRRD::ImageView<float> view(w, h, d, (float*)volume.data());
        vd = std::make_shared<VolumeRendering::VoxelData>(view, use_ess);
    }

    void set_model_transform(const Eigen::Matrix4d& model_matrix) {
        vd->setModelTransform(model_matrix);
    }

    Eigen::Matrix4d get_model_transform() const {
        return vd->getModelTransform();
    }

    void center_volume() {
        vd->centerVolume();
    }

    void empty_space_skipping(int bin_factor) {
        vd->emptySpaceSkipping(bin_factor);
    }

    std::shared_ptr<VolumeRendering::VoxelData> get_internal() const { return vd; }
};

class PyRaycaster {
    std::shared_ptr<VolumeRendering::Raycaster> rc;
    std::shared_ptr<PyVoxelData> vd_keep_alive;
public:
    PyRaycaster(std::shared_ptr<PyVoxelData> vd) : vd_keep_alive(vd) {
        rc = std::make_shared<VolumeRendering::Raycaster>(*(vd->get_internal()));
    }

    void set_samples_per_voxel(double samples) {
        rc->setSamplesPerVoxel(samples);
    }

    void set_clip_planes(const std::vector<Eigen::Vector4d>& planes) {
        rc->setClipPlanes(planes);
    }

    void set_raycast_pass(const std::string& type) {
        if (type == "Debug") {
            rc->raycastPass<VolumeRendering::Debug>();
        } else if (type == "MIP" || type == "MaximumIntensityProjection") {
            rc->raycastPass<VolumeRendering::MaximumIntensityProjection>();
        } else if (type == "IsoSurface" || type == "Iso") {
            rc->raycastPass<VolumeRendering::IsoSurface>();
        } else if (type == "DRR" || type == "DigitallyReconstructedRadiograph") {
            rc->raycastPass<VolumeRendering::DigitallyReconstructedRadiograph>();
        } else if (type == "EmissionAbsorption" || type == "EA") {
            rc->raycastPass<EmissionAbsorption>();
        } else if (type == "EmissionAbsorptionShaded" || type == "EAShaded") {
            rc->raycastPass<EmissionAbsorptionShaded>();
        } else {
            throw std::runtime_error("Unknown raycast pass type: " + type);
        }
    }

    void set_iso_value(float val) {
        auto& pass = rc->raycastPass<VolumeRendering::IsoSurface>();
        pass.setIsoValue(val);
    }

    void set_ray_length_weighted(bool weighted) {
        auto& pass = rc->raycastPass<VolumeRendering::DigitallyReconstructedRadiograph>();
        pass.setRayLangthWeighted(weighted);
    }

    void set_transfer_function_ea(py::array_t<float, py::array::c_style | py::array::forcecast> tf_data, float min_val = std::numeric_limits<float>::quiet_NaN(), float max_val = std::numeric_limits<float>::quiet_NaN()) {
        auto& pass = rc->raycastPass<EmissionAbsorption>();
        pass.setTransferFunction(tf_data);
        pass.setRange(min_val, max_val);
    }

    void set_transfer_function_ea_shaded(py::array_t<float, py::array::c_style | py::array::forcecast> tf_data, float min_val = std::numeric_limits<float>::quiet_NaN(), float max_val = std::numeric_limits<float>::quiet_NaN()) {
        auto& pass = rc->raycastPass<EmissionAbsorptionShaded>();
        pass.setTransferFunction(tf_data);
        pass.setRange(min_val, max_val);
    }

    py::array_t<float> render(const Eigen::Matrix<double, 3, 4>& P, int width, int height, int channels=1) {
        py::array_t<float> result;
        NRRD::ImageView<float> view;

        if (channels <= 1) {
            result = py::array_t<float>({height, width});
            view = NRRD::ImageView<float>(width, height, 1, (float*)result.mutable_data());
        } else {
            result = py::array_t<float>({height, width, channels});
            view = NRRD::ImageView<float>(channels, width, height, (float*)result.mutable_data());
        }

        rc->render(view, P);
        return result;
    }
};

void init_volume_rendering(py::module &m) {
    py::class_<PyVoxelData, std::shared_ptr<PyVoxelData>>(m, "VoxelData")
        .def(py::init<py::array_t<float>, bool>(), py::arg("volume"), py::arg("use_ess") = true)
        .def("set_model_transform", &PyVoxelData::set_model_transform)
        .def("get_model_transform", &PyVoxelData::get_model_transform)
        .def("center_volume", &PyVoxelData::center_volume)
        .def("empty_space_skipping", &PyVoxelData::empty_space_skipping);

    py::class_<PyRaycaster>(m, "Raycaster")
        .def(py::init<std::shared_ptr<PyVoxelData>>())
        .def("set_samples_per_voxel", &PyRaycaster::set_samples_per_voxel)
        .def("set_clip_planes", &PyRaycaster::set_clip_planes)
        .def("set_raycast_pass", &PyRaycaster::set_raycast_pass)
        .def("set_iso_value", &PyRaycaster::set_iso_value)
        .def("set_ray_length_weighted", &PyRaycaster::set_ray_length_weighted)
        .def("set_transfer_function_ea", &PyRaycaster::set_transfer_function_ea, py::arg("tf_data"), py::arg("min_val") = std::numeric_limits<float>::quiet_NaN(), py::arg("max_val") = std::numeric_limits<float>::quiet_NaN())
        .def("set_transfer_function_ea_shaded", &PyRaycaster::set_transfer_function_ea_shaded, py::arg("tf_data"), py::arg("min_val") = std::numeric_limits<float>::quiet_NaN(), py::arg("max_val") = std::numeric_limits<float>::quiet_NaN())
        .def("render", &PyRaycaster::render, py::arg("P"), py::arg("width"), py::arg("height"), py::arg("channels") = 1);
}
