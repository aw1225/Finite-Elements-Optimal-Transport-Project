#!/usr/bin/env python3
"""Human-readable collector for T5 v52 result directories."""
import csv, glob, json, os, sys

def one(pattern):
    hits=sorted(glob.glob(pattern)); return hits[-1] if hits else None

def read_rows(path):
    if not path: return []
    with open(path,newline='') as f: return list(csv.DictReader(f))

def read_one(path):
    rows=read_rows(path); return rows[-1] if rows else None

def main():
    root=sys.argv[1] if len(sys.argv)>1 else os.getcwd()
    summary=read_one(one(os.path.join(root,'*reference_fas_summary_with_map.csv')))
    if summary is None: summary=read_one(one(os.path.join(root,'*reference_fas_summary.csv')))
    audit=read_one(one(os.path.join(root,'*science_audit.csv')))
    cert=read_one(one(os.path.join(root,'*benchmark_certificate.csv')))
    trajectory=read_rows(one(os.path.join(root,'*fas_discrete_error_trajectory.csv')))
    reference=read_rows(one(os.path.join(root,'*reference_trace.csv')))
    payload={
        'result_dir':os.path.abspath(root),
        'certificate':cert,
        'summary':summary,
        'science_audit':audit,
        'trajectory':trajectory,
        'reference_trace':reference,
    }
    print(json.dumps(payload,indent=2,sort_keys=True))
    if cert:
        print('\nT5 geometry certificate')
        for k in [
            'certificate_pass','source_global_cells_firedrake','target_global_cells_firedrake',
            'target_parsed_connected_triangle_components','target_parsed_min_angle_degrees',
            'target_parsed_max_aspect_edge_over_altitude','target_parsed_measured_neck_width_physical',
            'target_parsed_neck_width_over_max_neck_triangle_diameter','neck_width_over_sqrt_2tau',
            'bridge_margin_unit','curvature_min_connected_arc_numerical','theorem_discontinuity_threshold_crossed',
        ]:
            if k in cert: print(f'{k}={cert[k]}')
    if summary:
        print('\nT5 solver headline')
        for k in [
            'cycles','reached_target','initial_residual_p','final_residual_p','geomean_residual_contraction',
            'initial_discrete_reference_error','final_discrete_reference_error','geomean_discrete_error_contraction',
            'reference_certified','reference_updates','reference_residual_p','reference_change_rel',
            'reference_tail_over_final_error','observed_heat_actions','expected_heat_actions','heat_count_match',
            'fas_vs_reference_barycentric_map_l2_physical','fas_vs_reference_lobe_probability_l2',
            'fas_lobe_mixing_mean','ref_lobe_mixing_mean','target_neck_mass_partition',
            'fas_barycentric_support_escape_mass','ref_barycentric_support_escape_mass',
            'fas_rms_conditional_spread_physical','ref_rms_conditional_spread_physical',
        ]:
            if k in summary: print(f'{k}={summary[k]}')
    if audit:
        print('\nT5 map/lobe audit')
        for k in [
            'full_residual_p','full_residual_q','source_marginal_l2_physical','target_marginal_l2_physical',
            'target_left_mass_partition','target_neck_mass_partition','target_right_mass_partition',
            'fas_mean_left_probability','fas_mean_neck_probability','fas_mean_right_probability',
            'fas_conditional_neck_mass_balance_error','fas_lobe_mixing_mean','fas_side_normalized_lobe_mixing_mean',
            'fas_lobe_mixing_source_mass_gt_0p25','fas_lobe_mixing_source_mass_gt_0p50',
            'fas_left_right_ambiguous_source_mass','fas_barycentric_support_escape_mass',
            'fas_barycentric_mapped_into_neck_mass','fas_barycentric_central_slab_mass',
            'fas_det_min','fas_det_max','fas_orientation_reversing_area_physical',
            'fas_rms_conditional_spread_physical','fas_vs_reference_barycentric_map_l2_physical',
            'fas_vs_reference_lobe_probability_l2','ref_lobe_mixing_mean','ref_barycentric_support_escape_mass',
            'ref_rms_conditional_spread_physical','ref_det_min','ref_orientation_reversing_area_physical',
            'fas_map_denominator_min','fas_map_denominator_nonfinite_count',
        ]:
            if k in audit: print(f'{k}={audit[k]}')

if __name__=='__main__': main()
