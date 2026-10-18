"""Reproducible M1 fixtures; run against an installed fusemesh wheel.

Usage: python benchmarks/m1_baseline.py > build/m1-benchmark.json
These are initial development measurements, not release performance targets.
"""
import json
import platform
from pathlib import Path
import subprocess
import sys
import time

import fusemesh
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))
from geometry_helpers import box, sphere


def main():
    report={'python':sys.version.split()[0], 'platform':platform.platform(),
            'build':fusemesh.build_info(), 'workloads':[]}
    report['gpu']=subprocess.check_output([
        'nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip()
    workloads=[('cube',[box()],.04),
               ('overlap_4_spheres',[sphere((x,y,0),.7,3) for x,y in [(-.35,-.35),(.35,-.35),(-.35,.35),(.35,.35)]],.06),
               ('containment',[box((0,0,0),(3,3,3)),box((1,1,1),(2,2,2))],.06)]
    for name,meshes,h in workloads:
        runs=[]
        for _ in range(4):
            start=time.perf_counter()
            result=fusemesh.reconstruct(meshes,h,backend="dense")
            runs.append({'total_ms':(time.perf_counter()-start)*1000, **result.metadata})
        report['workloads'].append({'name':name,'shape_count':len(meshes),
             'input_triangles':sum(len(f) for _,f in meshes),'voxel_size':h,
             'cold':runs[0],'warm_median_total_ms':float(np.median([r['total_ms'] for r in runs[1:]])),
             'warm_runs':runs[1:]})
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
