"""The performance gate must reject slower, larger, or different workloads."""
import copy
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('m3_performance',Path(__file__).resolve().parents[1]/'benchmarks/m3_performance.py')
benchmark=importlib.util.module_from_spec(spec);spec.loader.exec_module(benchmark)


def baseline():
    return dict(gpu='test GPU',workloads=[dict(name='solid',input_sha256='same input',
        options={'voxel_size':.1},median_ms=10.,peak_device_bytes=100000,vertices=10,triangles=16)])


def test_benchmark_accepts_expected_jitter():
    reference=baseline();report=copy.deepcopy(reference)
    report['workloads'][0]['median_ms']=16.
    assert benchmark.check_report(report,reference)==[]


@pytest.mark.parametrize('key,value',[
    ('median_ms',18.),('peak_device_bytes',120000),('triangles',18),
    ('input_sha256','different input'),('options',{'voxel_size':.2}),
])
def test_benchmark_detects_regressions_or_changed_workloads(key,value):
    reference=baseline();report=copy.deepcopy(reference)
    report['workloads'][0][key]=value
    assert benchmark.check_report(report,reference)


def test_benchmark_requires_same_hardware_and_workload_set():
    reference=baseline();report=copy.deepcopy(reference);report['gpu']='other GPU'
    with pytest.raises(ValueError,match='GPU/driver'):
        benchmark.check_report(report,reference)
    report=baseline();report['workloads']=[]
    assert benchmark.check_report(report,reference)
