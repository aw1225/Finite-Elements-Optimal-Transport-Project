#!/usr/bin/env python3
"""Static/provenance audit for T5 v52r1 exploratory amendment; does not require Firedrake."""
from __future__ import annotations
import ast, hashlib, json, re, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
DRIVER=ROOT/'cx3_t5_thin_neck_v52r1.py'
ASSETS=ROOT/'t5_assets_v52'
PARENT_SHA='f59b123df8a4902a04390c5b267b28a82007ba3963b9d0015fc80b5ad86ca269'
EXPECTED_ASSET_SHA={
 't5_source_disk_v52.msh':'b8f020e59538a42f17feb82824550b1c8daad71d06ecc76612ea8acd5ee9eb81',
 't5_target_thin_neck_v52.msh':'1269c95b276059f2f4938088820654e58bf4f8a3227ed18a614b5aece6522e56',
}
EXPECTED_AST={
'physical_to_unit_vector':'c3fdb587026a09d56a8e49393e714d9c75a2e52a5e30fcb09dfd3cab4ba517d0',
'unit_to_physical_vector':'69c554a83ef921ad5136c3659f0dbe4749bed2759125c3e643abd27a8948701d',
'build_bridge_hierarchy':'d12d7c19b727bde25efdd8a9693e7bcad15b5a6043f59f6f6bffa5db36b484a1',
'safe_log':'d16d36b8364f4f1bb64a085bf44b76523c57985239f9d72c9fd9e880b289ed68',
'clamp_expr':'98cb47df52e83a9e8b1ae35ef822f664ee2712a31f8f576efc3275c63333acca',
'write_rows':'8d781a4adfc4da06fa22d669544d6bcac11eef0ae90eb7a6b9277335fd3683fd',
'l2_norm':'9adc12f6b5e89a663dcd2656f3ba037efd33c4b2a9d25bf366be55a7cace4e82',
'linf_norm':'e90cb6b453239f529fb5f06fa289f62300c10553e171a9a5e5936c384c25904d',
'global_dof_max':'f11b0e2e9cf490eb4bc2d5a6b29245932f9fc7c33fdc4893c45e278e6e888219',
'global_dof_min':'761b79f430a889464b678341509693167c577cbf74c2f8ad3572e9c966c0dbb3',
'global_nonfinite_count':'133b7271d95a0a35ecd34083a788971d2b3b6e890635deeb6aeb9726a6cea34f',
'FASLevel':'23fa284e9b305643325423d3862cb09b5362ad181857850d3d4ba1dad991042d',
'hierarchy_level_objects':'1961332733bf311ed3c882c48f2f1fbeab0d55837a891805455a8d42bb81ae8d',
'hierarchy_rows':'628ad005c0ea5cab84f11b93af16a7ee3fd00043f25a65bd3f47f841c746df96',
'assert_requested_hierarchy_is_active':'eb687c807d13b58bcac3b12792bdf5b0f9e0659838644b0f01c58208a22db7a2',
'build_problem':'a05e11a77a5c060d22b6eb61761a2506e87c53576e7bffa38681fa71bedffe77',
'gauge_aligned_fields':'09b3758b2af89abc3306edc669f0ec7f53aff4b5a6fa5e78d1918f470504a197',
'gauge_aligned_distance':'d3d9411721132d72642512cfbd7bbaebe0047a6e4036568b69315a79e97ca3f4',
'reference_norm':'9c47e437052f996eee79a46fb0d8283d82e2b68bbdbbce08a7558ddf2535f9f0',
}

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def ast_hashes(path):
 tree=ast.parse(path.read_text())
 out={}
 for node in tree.body:
  name=getattr(node,'name',None)
  if name in EXPECTED_AST:
   dump=ast.dump(node,annotate_fields=True,include_attributes=False)
   out[name]=hashlib.sha256(dump.encode()).hexdigest()
 return out

def read_assignment(tree,name):
 for node in tree.body:
  if isinstance(node,ast.Assign):
   for target in node.targets:
    if isinstance(target,ast.Name) and target.id==name:
     try:return ast.literal_eval(node.value)
     except Exception:return None
 return None

def main():
 errors=[]; lines=[]
 compile(DRIVER.read_text(),str(DRIVER),'exec')
 for f in ['collect_t5_v52.py','static_audit_v52r1.py','generate_t5_mesh_assets_v52.py']:
  if (ROOT/f).exists(): compile((ROOT/f).read_text(),str(ROOT/f),'exec')
 lines.append('python_parse_and_compile=PASS')
 driver_sha=sha(DRIVER)
 lines.append(f'driver_sha256={driver_sha}')
 actual=ast_hashes(DRIVER)
 for name,expected in EXPECTED_AST.items():
  ok=actual.get(name)==expected
  lines.append(f'{name}_AST_identical_to_T4_v50={int(ok)}')
  if not ok: errors.append(f'AST mismatch: {name}: {actual.get(name)} != {expected}')
 tree=ast.parse(DRIVER.read_text())
 for name,want in [('N_SUB',16),('BRIDGE_WIDTH_PHYSICAL',4.0)]:
  got=read_assignment(tree,name); lines.append(f'{name}={got}')
  if got!=want: errors.append(f'{name} changed: {got} != {want}')
 text=DRIVER.read_text()
 required=['"ksp_type": "preonly"','"pc_type": "cholesky"','"pc_factor_mat_solver_type": "mumps"',
           'FAS_PRE_SMOOTH", "2"','FAS_COARSE_SMOOTH", "7"','FAS_POST_SMOOTH", "1"',
           'FAS_CORRECTION_DAMPING", "1.0"','FAS_COARSE_TAU_RATIO", "1.0"','FAS_TARGET_RESIDUAL", "4.31e-3"']
 for token in required:
  ok=token in text; lines.append(f'required_token_{hashlib.sha1(token.encode()).hexdigest()[:8]}={int(ok)}')
  if not ok: errors.append(f'missing frozen token: {token}')
 for name,expected in EXPECTED_ASSET_SHA.items():
  got=sha(ASSETS/name); ok=got==expected
  lines.append(f'{name}_sha256={got}'); lines.append(f'{name}_sha_match={int(ok)}')
  if not ok: errors.append(f'asset hash mismatch {name}')
 meta=json.loads((ASSETS/'t5_mesh_assets_v52_metadata.json').read_text())
 lines += [
  f'parent_t4_v50_sha256={PARENT_SHA}',
  f'target_triangle_count={meta["target"]["triangle_count"]}',
  f'target_connected_components={meta["target"]["connected_triangle_components"]}',
  f'target_min_angle_degrees={meta["target"]["min_angle_degrees"]}',
  f'target_max_aspect={meta["target"]["max_aspect_edge_over_altitude"]}',
  f'neck_width_physical={meta["target"]["neck_width_physical"]}',
  f'neck_width_over_max_neck_triangle_diameter={meta["target"]["neck_width_over_max_neck_triangle_diameter"]}',
  f'neck_width_over_sqrt_2tau={meta["neck_width_over_sqrt_2tau"]}',
  f'curvature_min_connected_arc_numerical={meta["target"]["minimum_connected_arc_curvature_integral_numerical"]}',
  f'theorem_discontinuity_threshold_crossed={meta["target"]["theorem_discontinuity_threshold_crossed"]}',
 ]
 # Wrapper contract checks.
 for f in ['t5_production_reference_B230_B460_B920_v52r1.pbs']:
  p=ROOT/f
  if not p.exists(): errors.append(f'missing wrapper {f}'); continue
  s=p.read_text()
  for token in ['module load Firedrake/2025.10.2-foss-2025b','mpiprocs=4','mpiexec -n 4 python3 -B -u','EXPECTED_SRC_SHA=', 'EXPECTED_SOURCE_MESH_SHA=', 'EXPECTED_TARGET_MESH_SHA=']:
   if token not in s: errors.append(f'{f}: missing {token}')
  m=re.search(r'EXPECTED_SRC_SHA="([0-9a-f]{64})"',s)
  if not m or m.group(1)!=driver_sha: errors.append(f'{f}: EXPECTED_SRC_SHA not synchronized with driver')
 lines.append(f'static_audit_pass={int(not errors)}')
 print('\n'.join(lines))
 if errors:
  print('\nERRORS:',file=sys.stderr)
  for e in errors: print('- '+e,file=sys.stderr)
  raise SystemExit(2)

if __name__=='__main__': main()
