import gc
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import fusemesh
from geometry_helpers import (box, combine, sphere, torus, box_sdf, sample_positions,
                              reference_field, unsigned_distance, mesh_quality, surface_samples)


@pytest.mark.gpu
@pytest.mark.parametrize('bounds,components', [
    ([((0,0,0),(1,1,1))],1),
    ([((0,0,0),(1,1,1)),((1.5,0,0),(2.5,1,1))],2),
    ([((0,0,0),(1,1,1)),((.4,.25,.2),(1.4,1.25,1.2))],1),
    ([((0,0,0),(2,2,2)),((.7,.7,.7),(1.2,1.2,1.2))],1),
])
def test_boxes_against_analytic_union(bounds,components):
    out=fusemesh.reconstruct([box(*b) for b in bounds],.1,return_field=True)
    points=sample_positions(out)
    reference=np.minimum.reduce([box_sdf(points,*b) for b in bounds])
    np.testing.assert_allclose(out.field,reference,atol=2e-8,rtol=1e-12)
    volumes=mesh_quality(out,components,2*components)
    assert np.all(volumes>0)
    exact=sum(np.prod(np.asarray(hi)-lo) for lo,hi in bounds)
    if len(bounds)==2:
        intersection=np.maximum(np.minimum(bounds[0][1],bounds[1][1])-np.maximum(bounds[0][0],bounds[1][0]),0)
        exact-=np.prod(intersection)
    assert abs(volumes.sum()-exact)/exact<.025
    boundary=surface_samples(out)
    errors=np.abs(np.minimum.reduce([box_sdf(boundary,*b) for b in bounds]))
    assert errors.max()<.08
    assert out.metadata['peak_device_bytes']<=512*1024**2
    assert all(t>=0 for t in out.metadata['timings_ms'].values())


@pytest.mark.gpu
def test_containment_and_coincident_meshes_do_not_add_surfaces():
    outer=box((0,0,0),(3,3,3))
    baseline=fusemesh.reconstruct([outer],.15)
    for meshes in ([outer,box((1,1,1),(2,2,2))],[outer,outer],[outer,(outer[0],outer[1][:,::-1])]):
        result=fusemesh.reconstruct(meshes,.15)
        np.testing.assert_array_equal(result.faces,baseline.faces)
        np.testing.assert_allclose(result.vertices,baseline.vertices,atol=1e-12)
        mesh_quality(result,1,2)


@pytest.mark.gpu
def test_enclosed_cavity_is_preserved():
    inner=box((1,1,1),(2,2,2))
    shell=combine(box((0,0,0),(3,3,3)),(inner[0],inner[1][:,::-1]))
    result=fusemesh.reconstruct([shell],.1,return_field=True)
    volumes=np.sort(mesh_quality(result,2,4))
    assert volumes[0]<0<volumes[1]
    assert abs(volumes.sum()-26)<.15
    p=sample_positions(result)
    reference=np.maximum(box_sdf(p,(0,0,0),(3,3,3)),-box_sdf(p,(1,1,1),(2,2,2)))
    np.testing.assert_allclose(result.field,reference,atol=2e-8)


@pytest.mark.gpu
def test_ray_vertex_edge_ties_against_independent_solid_angles():
    # With h=.4, the grid contains y=z=0. +X rays meet octahedron
    # vertices exactly; diagonal face-edge rays are also present.
    meshes=[sphere(levels=0)]
    result=fusemesh.reconstruct(meshes,.4,return_field=True)
    p=sample_positions(result).reshape(-1,3)
    expected=reference_field(p,meshes)
    np.testing.assert_allclose(result.field.ravel(),expected,atol=8e-8)
    mesh_quality(result,1,2)


@pytest.mark.gpu
def test_rotated_overlapping_solids_against_independent_reference():
    v,f=box((-1,-.6,-.4),(1,.6,.4))
    angle=.37
    rotation=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
    meshes=[(v@rotation.T,f),sphere((.7,.1,.1),.6,2)]
    out=fusemesh.reconstruct(meshes,.12,return_field=True)
    points=sample_positions(out).reshape(-1,3)
    rng=np.random.default_rng(81)
    selected=rng.choice(len(points),180,replace=False)
    expected=reference_field(points[selected],meshes)
    np.testing.assert_allclose(out.field.ravel()[selected],expected,atol=3e-8)
    mesh_quality(out,1,2)
    reordered=fusemesh.reconstruct(meshes[::-1],.12)
    np.testing.assert_array_equal(out.faces,reordered.faces)
    np.testing.assert_array_equal(out.vertices,reordered.vertices)


@pytest.mark.gpu
def test_sphere_bidirectional_surface_error_and_resolution():
    mesh=sphere(levels=4)
    coarse=fusemesh.reconstruct([mesh],.22)
    fine=fusemesh.reconstruct([mesh],.09)
    coarse_volume=mesh_quality(coarse,1,2).sum()
    fine_volume=mesh_quality(fine,1,2).sum()
    analytic_volume=4*np.pi/3
    assert abs(fine_volume-analytic_volume)<abs(coarse_volume-analytic_volume)
    assert abs(fine_volume-analytic_volume)/analytic_volume<.025
    # Sample triangle interiors as well as output vertices.
    errors=np.abs(np.linalg.norm(surface_samples(fine),axis=1)-1)
    assert errors.max()<.015
    directions=np.random.default_rng(2).normal(size=(120,3))
    directions/=np.linalg.norm(directions,axis=1)[:,None]
    distances=unsigned_distance(directions,fine.vertices,fine.faces.astype(np.int64))
    assert distances.max()<.015


@pytest.mark.gpu
def test_torus_preserves_resolved_hole():
    out=fusemesh.reconstruct([torus()],.09)
    volume=mesh_quality(out,1,0).sum()
    assert abs(volume-2*np.pi**2*.35**2)/(2*np.pi**2*.35**2)<.06
    assert np.linalg.norm(out.vertices[:,:2],axis=1).min()>.6


@pytest.mark.gpu
def test_large_coordinate_offset():
    offset=np.array([1e8,-2e8,3e8])
    v,f=box()
    base=fusemesh.reconstruct([(v,f)],.1)
    shifted=fusemesh.reconstruct([(v+offset,f)],.1)
    np.testing.assert_array_equal(base.faces,shifted.faces)
    np.testing.assert_allclose(shifted.vertices-offset,base.vertices,atol=5e-8)
    assert abs(mesh_quality(shifted,1,2).sum()-1)<.025


@pytest.mark.gpu
def test_subvoxel_feature_loss_is_explicit():
    meshes=[box(),box((1.2,.21,.21),(1.22,.23,.23))]
    coarse=fusemesh.reconstruct(meshes,.1)
    fine=fusemesh.reconstruct(meshes,.01,max_memory_bytes=256*1024**2)
    mesh_quality(coarse,1,2)
    mesh_quality(fine,2,4)


@pytest.mark.gpu
@pytest.mark.parametrize('offset',[(1,0,0),(1,1,0),(1,1,1),(1-1e-8,0,0)])
def test_contacts_are_finite_and_deterministic(offset):
    v,f=box()
    meshes=[(v,f),(v+offset,f)]
    a=fusemesh.reconstruct(meshes,.13)
    b=fusemesh.reconstruct(meshes,.13)
    assert np.isfinite(a.vertices).all()
    assert a.faces.max()<len(a.vertices)
    np.testing.assert_array_equal(a.faces,b.faces)
    np.testing.assert_array_equal(a.vertices,b.vertices)
    # No manifold or exact-contact topology promise for degenerate contacts.


@pytest.mark.gpu
def test_resource_budget_enforced_and_failure_does_not_poison_device():
    mesh=box()
    a=fusemesh.reconstruct([mesh],.1)
    with pytest.raises(fusemesh.ResourceError,match='max_memory_bytes'):
        fusemesh.reconstruct([mesh],.1,max_memory_bytes=a.metadata['peak_device_bytes']-1)
    b=fusemesh.reconstruct([mesh],.1)
    np.testing.assert_array_equal(a.faces,b.faces)
    np.testing.assert_array_equal(a.vertices,b.vertices)


@pytest.mark.gpu
def test_dtype_strides_ownership_and_concurrent_calls():
    v,f=box()
    wide=np.zeros((8,6),dtype=np.float32);wide[:,::2]=v
    result=fusemesh.reconstruct([(wide[:,::2],f.astype(np.int32))],.2)
    assert result.vertices.dtype==np.float64 and result.faces.dtype==np.uint64
    saved=result.vertices.copy();vertices=result.vertices
    del result;gc.collect()
    np.testing.assert_array_equal(vertices,saved)
    with ThreadPoolExecutor(max_workers=3) as pool:
        results=list(pool.map(lambda _: fusemesh.reconstruct([(v,f)],.2),range(6)))
    for other in results:
        np.testing.assert_array_equal(other.vertices,saved)


def test_empty_input_without_cuda():
    out=fusemesh.reconstruct([],.1,return_field=True)
    assert out.vertices.shape==(0,3) and out.faces.shape==(0,3)
    assert out.field.shape==(0,0,0) and out.metadata['peak_device_bytes']==0
    code="import fusemesh; r=fusemesh.reconstruct([], .1); assert r.vertices.shape==(0,3)"
    subprocess.run([sys.executable,'-c',code],env={**os.environ,'CUDA_VISIBLE_DEVICES':''},check=True)


@pytest.mark.parametrize('spacing',[0,-1,np.nan,np.inf])
def test_invalid_spacing(spacing):
    with pytest.raises(ValueError,match='voxel_size'):
        fusemesh.reconstruct([],spacing)


def test_structural_and_topology_errors():
    v,f=box()
    cases=[((v[:,0],f),'vertices must have shape'),
           ((v,f.astype(float)),'integer dtype'),
           ((v,f-1),'nonnegative'),
           ((v,np.array([[0,1,99]])),'outside the vertex array'),
           ((v,np.array([[0,0,1]])),'zero area'),
           ((v,f[:-1]),'open'),
           ((v,np.concatenate((f,f[:1]))),'nonmanifold')]
    bad=f.copy();bad[0]=bad[0,::-1];cases.append(((v,bad),'orientation'))
    nan=v.copy();nan[0,0]=np.nan;cases.append(((nan,f),'finite'))
    for mesh,message in cases:
        with pytest.raises((ValueError,TypeError),match=message):
            fusemesh.reconstruct([mesh],.1)
    with pytest.raises(ValueError,match='validation'):
        fusemesh.reconstruct([],1,validation='repair')
    with pytest.raises(ValueError,match='nonnegative'):
        fusemesh.reconstruct([],1,device=-1)
    with pytest.raises(ValueError,match='max_memory_bytes'):
        fusemesh.reconstruct([],1,max_memory_bytes=0)
    with pytest.raises(fusemesh.ResourceError,match='dense grid'):
        fusemesh.reconstruct([box()],1e-8)


def test_nonmanifold_vertex_link_is_rejected():
    v,f=box()
    # Two boxes touch at a single indexed vertex, with otherwise separate indices.
    v2=v+1
    joined=np.concatenate((v,v2[1:]))
    mapping=np.arange(8)+7;mapping[0]=6
    faces=np.concatenate((f,mapping[f]))
    with pytest.raises(ValueError,match='vertex link'):
        fusemesh.reconstruct([(joined,faces)],.2)


@pytest.mark.gpu
def test_thin_wall_and_narrow_gap_depend_on_resolution():
    wall=box((0,0,0),(1,1,.06))
    coarse=fusemesh.reconstruct([wall],.2)
    assert coarse.vertices.shape==(0,3) and coarse.faces.shape==(0,3)
    mesh_quality(fusemesh.reconstruct([wall],.02),1,2)
    separated=[box(),box((1.08,0,0),(2.08,1,1))]
    mesh_quality(fusemesh.reconstruct(separated,.2),1,2)
    mesh_quality(fusemesh.reconstruct(separated,.04),2,4)


@pytest.mark.gpu
def test_basic_validation_and_unused_vertices_preserve_geometry():
    v,f=box()
    with_unused=np.concatenate((v,[[1e100,1e100,1e100]]))
    full=fusemesh.reconstruct([(with_unused,f)],.2)
    basic=fusemesh.reconstruct([(v,f)],.2,validation='basic')
    np.testing.assert_array_equal(full.faces,basic.faces)
    np.testing.assert_array_equal(full.vertices,basic.vertices)


def test_empty_mesh_and_reconstruction_without_visible_gpu():
    mesh=(np.empty((0,3)),np.empty((0,3),dtype=np.int64))
    assert fusemesh.reconstruct([mesh],.1).faces.shape==(0,3)
    code="""
import numpy as np
import fusemesh
v=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=float)
f=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]])
fusemesh.reconstruct([(v,f)], .2)
"""
    result=subprocess.run([sys.executable,'-c',code],env={**os.environ,'CUDA_VISIBLE_DEVICES':''},capture_output=True,text=True)
    assert result.returncode!=0 and 'RuntimeError: FUSE CUDA:' in result.stderr


@pytest.mark.gpu
@pytest.mark.parametrize('scale',[2.0**-300,2.0**300])
def test_world_unit_scale_invariance(scale):
    # A world-space squared area underflows/overflows at these scales, although
    # grid geometry and all input/output coordinates are well represented.
    v,f=box()
    base=fusemesh.reconstruct([(v,f)],.125)
    scaled=fusemesh.reconstruct([(v*scale,f)],.125*scale)
    np.testing.assert_array_equal(base.faces,scaled.faces)
    np.testing.assert_allclose(scaled.vertices/scale,base.vertices,atol=1e-14)
