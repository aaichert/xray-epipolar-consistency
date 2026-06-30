#include <pybind11/pybind11.h>
#include <string>


#include "HeaderOnly/NRRD/nrrd.hxx"

namespace py = pybind11;



#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "LibUtilsCuda/CudaBindlessTexture.h"
#include "LibEpipolarConsistency/EpipolarConsistencyRadonIntermediateCPU.hxx"
#include "LibEpipolarConsistency/EpipolarConsistencyRadonIntermediate.h"
#include "LibEpipolarConsistency/RadonIntermediate.h"

class PyRadonIntermediate {
    std::shared_ptr<EpipolarConsistency::RadonIntermediate> ri;
public:
    PyRadonIntermediate(py::array_t<float, py::array::c_style | py::array::forcecast> projectionData,
                        int size_alpha, int size_t, int filter, int post_process) {
        int w = projectionData.shape(1);
        int h = projectionData.shape(0);
        NRRD::ImageView<float> view(w, h, 1, (float*)projectionData.data());
        ri = std::make_shared<EpipolarConsistency::RadonIntermediate>(
            view, size_alpha, size_t, 
            (EpipolarConsistency::RadonIntermediate::Filter)filter, 
            (EpipolarConsistency::RadonIntermediate::PostProcess)post_process);
    }
    
    py::array_t<float> get_data() {
        ri->readback(false); // Ensure CPU data is up to date
        auto& data = ri->data(); // NRRD::ImageView<float>
        int w = data.size(0);
        int h = data.size(1);
        auto result = py::array_t<float>({h, w});
        std::memcpy(result.mutable_data(), (const float*)data, w * h * sizeof(float));
        return result;
    }

    std::shared_ptr<EpipolarConsistency::RadonIntermediate> get_internal() const { return ri; }
};

// Wrapper for computeForImagePair using CPU-based RadonIntermediate
py::tuple compute_for_image_pair_wrapper(
    const Geometry::ProjectionMatrix& P0, 
    const Geometry::ProjectionMatrix& P1,
    const PyRadonIntermediate& dtr0,
    const PyRadonIntermediate& dtr1,
    int num_planes,
    double object_radius_mm) 
{
    Eigen::Vector4d C0 = Geometry::getCameraCenter(P0);
    Eigen::Vector4d C1 = Geometry::getCameraCenter(P1);

    dtr0.get_internal()->readback();
    dtr1.get_internal()->readback();

    std::vector<double> samples0, samples1, angles;
    double weight = 1.0;
    double cost = EpipolarConsistency::MetricCPU::computeForImagePair(
        C0, C1, P0, P1, 
        dtr0.get_internal().get(), 
        dtr1.get_internal().get(), 
        num_planes, 
        object_radius_mm,
        nullptr, 
        &samples0, 
        &samples1, 
        &angles,
        nullptr,
        &weight
    );

    auto v0s_arr = py::array_t<float>(samples0.size());
    float* v0s_ptr = v0s_arr.mutable_data();
    for (size_t i = 0; i < samples0.size(); ++i) {
        v0s_ptr[i] = (float)samples0[i];
    }

    auto v1s_arr = py::array_t<float>(samples1.size());
    float* v1s_ptr = v1s_arr.mutable_data();
    for (size_t i = 0; i < samples1.size(); ++i) {
        v1s_ptr[i] = (float)samples1[i];
    }

    auto kappas_arr = py::array_t<float>(angles.size());
    float* kappas_ptr = kappas_arr.mutable_data();
    for (size_t i = 0; i < angles.size(); ++i) {
        kappas_ptr[i] = (float)angles[i];
    }

    return py::make_tuple(cost, v0s_arr, v1s_arr, kappas_arr, weight);
}


class PyMetricRadonIntermediate {
    std::unique_ptr<EpipolarConsistency::MetricRadonIntermediate> metric;
    std::vector<py::object> dtrs_keep_alive;
public:
    PyMetricRadonIntermediate() : metric(std::make_unique<EpipolarConsistency::MetricRadonIntermediate>()) {}

    void setRadonIntermediates(const std::vector<py::object>& py_dtrs) {
        dtrs_keep_alive = py_dtrs;
        std::vector<EpipolarConsistency::RadonIntermediate*> raw_dtrs;
        for (auto& obj : py_dtrs) {
            auto& py_ri = obj.cast<PyRadonIntermediate&>();
            raw_dtrs.push_back(py_ri.get_internal().get());
        }
        metric->setRadonIntermediates(raw_dtrs);
    }

    void setProjectionMatrices(const std::vector<Geometry::ProjectionMatrix>& Ps) {
        metric->setProjectionMatrices(Ps);
    }

    void setObjectRadius(double radius_mm) {
        metric->setObjectRadius(radius_mm);
    }

    void setEpipolarPlaneNumber(int num_planes) {
        metric->setEpipolarPlaneNumber(num_planes);
    }

    py::array_t<float> evaluate_indices(const std::vector<Eigen::Vector4i>& indices) {
        int n_pairs = indices.size();
        auto result = py::array_t<float>(n_pairs);
        metric->evaluate(indices, result.mutable_data());
        return result;
    }

    double evaluate() {
        return metric->evaluate(nullptr);
    }

    py::array_t<float> evaluate_pairwise() {
        int n = metric->getNumberOfProjetions();
        auto result = py::array_t<float>({n, n});
        float* ptr = result.mutable_data();
        for (int i = 0; i < n * n; ++i) {
            ptr[i] = 0.0f;
        }
        metric->evaluate(ptr);
        return result;
    }

    py::tuple getRedundantSignalsForViews(int i, int j) {
        std::vector<float> v0s, v1s, kappas;
        double weight = 1.0;
        double cost = metric->evaluateForImagePair(i, j, &v0s, &v1s, &kappas, &weight);

        auto v0s_arr = py::array_t<float>(v0s.size());
        std::memcpy(v0s_arr.mutable_data(), v0s.data(), v0s.size() * sizeof(float));

        auto v1s_arr = py::array_t<float>(v1s.size());
        std::memcpy(v1s_arr.mutable_data(), v1s.data(), v1s.size() * sizeof(float));

        auto kappas_arr = py::array_t<float>(kappas.size());
        std::memcpy(kappas_arr.mutable_data(), kappas.data(), kappas.size() * sizeof(float));

        return py::make_tuple(cost, v0s_arr, v1s_arr, kappas_arr, weight);
    }

    py::array_t<float> compute_zero_plane_distances() {
        int n = metric->getNumberOfProjetions();
        auto result = py::array_t<float>({n, n});
        float* ptr = result.mutable_data();
        for (int i = 0; i < n * n; ++i) {
            ptr[i] = 0.0f;
        }

        auto& dtrs = metric->getRadonIntermediates();
        if (dtrs.empty() || n == 0) return result;
        float n_u = dtrs[0]->getOriginalImageSize(0);
        float n_v = dtrs[0]->getOriginalImageSize(1);
        float n_x2 = n_u * 0.5f;
        float n_y2 = n_v * 0.5f;

        auto& Ps = metric->getProjectionMatrices();

        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                Eigen::Vector4f C0 = Geometry::getCameraCenter(Ps[i]).cast<float>().eval();
                Eigen::Vector4f C1 = Geometry::getCameraCenter(Ps[j]).cast<float>().eval();
                Eigen::Matrix<float, 3, 4> P0invT = Geometry::pseudoInverse(Ps[i]).transpose().cast<float>().eval();
                Eigen::Matrix<float, 3, 4> P1invT = Geometry::pseudoInverse(Ps[j]).transpose().cast<float>().eval();

                float K0[8], K1[8];
                computeK01(n_x2, n_y2, C0.data(), C1.data(), P0invT.data(), P1invT.data(), 0.0f, 1, K0, K1);

                ptr[j * n + i] = std::abs(K0[2]);
                ptr[i * n + j] = std::abs(K1[2]);
            }
        }
        return result;
    }
};


#include "LibEpipolarConsistency/EpipolarConsistencyCommon.hxx"

py::tuple py_computeK01(float n_x2, float n_y2, py::array_t<float, py::array::c_style | py::array::forcecast> C0, py::array_t<float, py::array::c_style | py::array::forcecast> C1, py::array_t<float, py::array::c_style | py::array::forcecast> P0invT, py::array_t<float, py::array::c_style | py::array::forcecast> P1invT, float object_radius_mm, int num_planes) {
    // Eigen transpose to ensure column-major format as expected by computeK01
    Eigen::Map<const Eigen::Matrix<float, 3, 4, Eigen::RowMajor>> p0_map((float*)P0invT.data());
    Eigen::Map<const Eigen::Matrix<float, 3, 4, Eigen::RowMajor>> p1_map((float*)P1invT.data());
    Eigen::Matrix<float, 3, 4, Eigen::ColMajor> p0_col = p0_map;
    Eigen::Matrix<float, 3, 4, Eigen::ColMajor> p1_col = p1_map;

    float K0[8], K1[8];
    computeK01(n_x2, n_y2, (float*)C0.data(), (float*)C1.data(), (float*)p0_col.data(), (float*)p1_col.data(), object_radius_mm, num_planes, K0, K1);

    auto py_K0 = py::array_t<float>(8);
    auto py_K1 = py::array_t<float>(8);
    std::memcpy(py_K0.mutable_data(), K0, 8 * sizeof(float));
    std::memcpy(py_K1.mutable_data(), K1, 8 * sizeof(float));
    return py::make_tuple(py_K0, py_K1);
}

py::tuple py_lineToSampleDtr(py::array_t<float, py::array::c_style | py::array::forcecast> line, float range_t) {
    float l[3];
    std::memcpy(l, line.data(), 3 * sizeof(float));
    bool res = lineToSampleDtr(l, range_t);
    return py::make_tuple(l[0], l[1], res);
}



void init_volume_rendering(py::module &m);

PYBIND11_MODULE(_core, m) {
    m.doc() = "Epipolar Consistency Core Module";
    init_volume_rendering(m);

    m.def("compute_for_image_pair", &compute_for_image_pair_wrapper,
          py::arg("P0"), py::arg("P1"), py::arg("dtr0"), py::arg("dtr1"),
          py::arg("num_planes") = 1800, py::arg("object_radius_mm") = 0.0,
          "Compute Epipolar Consistency for image pair");

    py::enum_<EpipolarConsistency::RadonIntermediate::Filter>(m, "RadonFilter")
        .value("Derivative", EpipolarConsistency::RadonIntermediate::Filter::Derivative)
        .value("Ramp", EpipolarConsistency::RadonIntermediate::Filter::Ramp)
        .value("None", EpipolarConsistency::RadonIntermediate::Filter::None)
        .export_values();

    py::enum_<EpipolarConsistency::RadonIntermediate::PostProcess>(m, "RadonPostProcess")
        .value("Identity", EpipolarConsistency::RadonIntermediate::PostProcess::Identity)
        .value("SquareRoot", EpipolarConsistency::RadonIntermediate::PostProcess::SquareRoot)
        .value("Logarithm", EpipolarConsistency::RadonIntermediate::PostProcess::Logarithm)
        .export_values();

    py::class_<PyRadonIntermediate>(m, "RadonIntermediate")
        .def(py::init<py::array_t<float>, int, int, int, int>())
        .def("get_data", &PyRadonIntermediate::get_data);

    py::class_<PyMetricRadonIntermediate>(m, "MetricRadonIntermediate")
        .def(py::init<>())
        .def("setRadonIntermediates", &PyMetricRadonIntermediate::setRadonIntermediates)
        .def("setProjectionMatrices", &PyMetricRadonIntermediate::setProjectionMatrices)
        .def("setObjectRadius", &PyMetricRadonIntermediate::setObjectRadius)
        .def("setEpipolarPlaneNumber", &PyMetricRadonIntermediate::setEpipolarPlaneNumber)
        .def("evaluate", py::overload_cast<>(&PyMetricRadonIntermediate::evaluate))
        .def("evaluate_pairwise", &PyMetricRadonIntermediate::evaluate_pairwise)
        .def("evaluate_indices", &PyMetricRadonIntermediate::evaluate_indices)
        .def("getRedundantSignalsForViews", &PyMetricRadonIntermediate::getRedundantSignalsForViews)
        .def("compute_zero_plane_distances", &PyMetricRadonIntermediate::compute_zero_plane_distances);

    m.def("computeK01", &py_computeK01, "Python wrapper for EpipolarConsistencyCommon.hxx computeK01");
    m.def("lineToSampleDtr", &py_lineToSampleDtr, "Python wrapper for EpipolarConsistencyCommon.hxx lineToSampleDtr");
    

}
