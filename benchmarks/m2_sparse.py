"""Compare M2 sparse and M1 dense backends on the same installed wheel."""
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np
import fusemesh

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
from geometry_helpers import box, sphere


def main():
    report = dict(python=sys.version.split()[0],platform=platform.platform(),
                  build=fusemesh.build_info(),version=fusemesh.__version__,workloads=[])
    report['gpu'] = subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip()
    workloads = [
        ('small_cube',[box()],.04,True),
        ('large_cube',[box((0,0,0),(4,4,4))],.05,True),
        ('overlap_spheres',[sphere((x,y,0),.7,3) for x,y in [(-.35,-.35),(.35,-.35),(-.35,.35),(.35,.35)]],.06,True),
        ('containment',[box((0,0,0),(16,16,16)),box((7,7,7),(9,9,9))],.25,True),
        ('distant_cubes',[box(),box((4096,4096,4096),(4097,4097,4097))],.25,False),
    ]
    for name,meshes,h,run_dense in workloads:
        entry = dict(name=name,voxel_size=h,input_triangles=sum(len(f) for _,f in meshes),backends={})
        outputs = {}
        for backend in (['dense','sparse'] if run_dense else ['sparse']):
            runs = []
            for _ in range(4):
                start = time.perf_counter()
                result = fusemesh.reconstruct(meshes,h,backend=backend)
                runs.append(dict(total_ms=(time.perf_counter()-start)*1000,**result.metadata))
            entry['backends'][backend] = dict(first=runs[0],warm_runs=runs[1:],warm_median_ms=float(np.median([r['total_ms'] for r in runs[1:]])))
            outputs[backend] = result
        if run_dense:
            np.testing.assert_array_equal(outputs['dense'].vertices,outputs['sparse'].vertices)
            np.testing.assert_array_equal(outputs['dense'].faces,outputs['sparse'].faces)
            entry['exact_indexed_equivalence'] = True
        report['workloads'].append(entry)
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
