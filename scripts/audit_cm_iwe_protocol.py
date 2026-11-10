#!/usr/bin/env python3
"""Freeze and replay the accepted CM-IWE evaluation, with a baseline gate.

Run from the repository root using python -m scripts.audit_cm_iwe_protocol.
Original result files and event caches are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from ami_simulator.event_quality import compute_reference_edge_mask
from ami_simulator.pipeline import load_rgb_sequence, rgb_to_linear_luminance
from scripts.evaluate_cm_edge_reconstruction import _cm_edges, _edge_f1
from scripts.plot_paper_dvs_v2e_cm_coverage import (
    CACHE_WINDOWS, CLIP_CACHE, READY, VELOCITY_GRID, _scene_path, _timestamps,
)

ORIGINAL = Path('results/ami_next/cm_edge_reconstruction_all_regimes.json')
OUT = Path('results/ami_next/cm_iwe_protocol_matched')
STUDENT = Path('results/eveyes_v2_regime_f1_argmax_m0/policy.pt')
KEYS = ('edge_f1', 'edge_precision', 'edge_coverage', 'cm_contrast')
REQUESTED = ('camera_only', 'object_only', 'co_motion', 'counter_motion')


def read(path):
    return json.loads(Path(path).read_text())


def save(name, data):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def specs():
    return [{k: row[k] for k in ('split', 'scene', 'regime', 'hero')}
            for row in read(ORIGINAL)['rows'] if row['method'] == 'ami_fixed_180']


def freeze():
    if (OUT / 'protocol_summary.json').exists():
        raise SystemExit('Protocol already frozen; use baseline or adaptive stage.')
    windows = []
    for spec in specs():
        scene = _scene_path(READY, spec['split'], spec['scene'], spec['regime'])
        h = spec['hero']
        times = _timestamps(scene)
        # Use the original loader and luminance conversion without resizing.
        frames, _ = load_rgb_sequence(scene / 'images', scene / 'timestamps.csv', timestamp_unit='s', max_frames=h + 2)
        reference = compute_reference_edge_mask(rgb_to_linear_luminance(frames[h:h+1])[0], edge_quantile=.75)
        paths = sorted(p for p in (scene / 'images').iterdir() if p.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.tif','.tiff'})
        sources = {}
        for method in read(ORIGINAL)['aggregate']:
            if method.startswith('ami_fixed'):
                rpm = int(method.rsplit('_', 1)[1])
                path = CACHE_WINDOWS / f'{scene.parent.name}_{scene.name}_f{h}_rpm{rpm}_reference_n1.npz'
                mode = 'reference-plane cached window'
            else:
                path = CLIP_CACHE / f"window_{spec['split']}_{spec['scene']}_{spec['regime']}_f{h}_{method}.npz"
                mode = 'cached window'
                if not path.exists():
                    path = CLIP_CACHE / f"{spec['split']}_{spec['scene']}_{spec['regime']}_f{h}_w48_{method}.npz"
                    mode = '48-frame warmup clip; slice (t0,t1]'
            if not path.exists():
                raise FileNotFoundError(f'Required original event input missing: {path}')
            sources[method] = {'path': str(path), 'sha256': sha(path), 'mode': mode}
        windows.append({**spec, 'scene_path': str(scene), 't0': float(times[h]), 't1': float(times[h+1]),
                        'duration_s': float(times[h+1]-times[h]), 'rgb_path': str(paths[h]),
                        'rgb_sha256': sha(paths[h]), 'reference_edge_sha256': array_sha(reference),
                        'reference_edge_pixels': int(reference.sum()), 'image_shape_hw': list(reference.shape),
                        'valid_mask_sha256': array_sha(np.ones(reference.shape, dtype=bool)),
                        'baseline_event_sources': sources})
        print('frozen', spec, flush=True)
    source_files = [
        'scripts/evaluate_cm_edge_reconstruction.py', 'scripts/plot_paper_dvs_v2e_cm_coverage.py',
        'scripts/plot_eveyes_event_qualitative.py', 'scripts/evaluate_dl3dv_downstream.py',
        'scripts/evaluate_inverse_motion_control.py', 'scripts/evaluate_phase_continuous_control.py',
        'scripts/evaluate_eveyes_v2_downstream.py', 'ami_simulator/contrast_maximization.py',
        'ami_simulator/pipeline.py', 'ami_simulator/event_quality.py', 'ami_simulator/warp.py',
        'ami_simulator/events.py', 'ami_simulator/geometry.py', 'scripts/train_inverse_motion_policy.py',
    ]
    command = [sys.executable, '-m', 'scripts.evaluate_cm_edge_reconstruction', '--dataset-root', str(READY),
               '--device', 'cpu', '--max-events', '80000', '--output', str(OUT / 'baseline_reproduction.json')]
    protocol = {
        'protocol': read(ORIGINAL)['protocol'], 'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'accepted_result': {'path': str(ORIGINAL), 'sha256': sha(ORIGINAL)},
        'source_sha256': {p: sha(p) for p in source_files},
        'reproduction_command': command,
        'historical_cli_provenance': 'Literal original shell command not saved in located reports. Effective arguments recovered from code defaults, result metadata and exact baseline reproduction.',
        'baseline_reproduction_scope': 'Re-score original frozen event NPZ inputs; do not regenerate DVS/v2e or fixed-AMI acquisition.',
        'requested_regimes': list(REQUESTED), 'evaluated_regimes': sorted({s['regime'] for s in windows}),
        'windows_per_regime': dict(Counter(s['regime'] for s in windows)), 'windows_per_method': len(windows),
        'requested_four_regime_result_available': False,
        'scope_discrepancy': 'Accepted table has drive_stop_drive, NOT counter_motion; no counter_motion window exists in this protocol. Train/valid/test are mixed representative windows, not an all-held-out benchmark.',
        'windows': windows, 'timestamp_unit': 's', 'event_window': '(t0,t1]',
        'rgb_reference': {'frame': 'hero at t0 (CM warp reference remains t1, as in original)',
                          'luminance': 'sRGB float32 -> linear RGB; 0.2126 R + 0.7152 G + 0.0722 B',
                          'edges': 'float64 log(max(luminance,1e-6)); np.gradient; hypot; 0.75 quantile over magnitudes >1e-12; >=threshold and >1e-12',
                          'empty_positive_gradient': 'all-false reference mask'},
        'valid_region': 'Full native 540x960 image, all-true mask, no crop, target mask or dilation; discard warped event coordinates outside image.',
        'compensation': 'AMI: per-event prism inverse mapping at event timestamp, continuous phase; np.rint reference coordinates. DVS/v2e: unmodulated reference plane (identity optical compensation).',
        'cm_motion_model': 'Global constant 2D translation, xprime=rint(x-vx*(t-t1)), yprime=rint(y-vy*(t-t1))',
        'velocity_grid': {'unit': 'pixels/s', 'count': len(VELOCITY_GRID), 'ordered_pairs': VELOCITY_GRID,
                          'vx': sorted({v[0] for v in VELOCITY_GRID}), 'vy': sorted({v[1] for v in VELOCITY_GRID}),
                          'step': 'nonuniform; exact lists above; vx outer loop, vy inner loop'},
        'polarity': 'Two unsigned count channels (p>0, p<=0); sum channel variances to select velocity; sum channel counts before edge extraction.',
        'event_filtering': 'No denoising or event-count normalization. If N>80000 select np.linspace(0,N-1,80000).astype(int64) from original event ordering; otherwise keep all events.',
        'cm_edges': 'np.gradient(log1p(sum-polarity IWE)); hypot; threshold at 0.75 quantile of magnitudes >1e-12; prediction >=threshold; empty positive set -> threshold infinity.',
        'edge_matching_tolerance_pixels': 0,
        'metrics': {'edge_precision': '|predicted & reference| / |predicted|, zero if denominator zero',
                    'edge_coverage': '|predicted & reference| / |reference|, zero if denominator zero',
                    'edge_f1': '2*precision*coverage/(precision+coverage), zero if denominator zero',
                    'cm_contrast': 'max over fixed grid of var(IWE_positive)+var(IWE_negative); float64 np.var ddof=0 over full image, diagnostic only; first candidate wins ties'},
        'aggregation': {'level': 'metrics per window first; arithmetic macro mean over all 8 windows per method',
                        'formula': 'M_method = (1/8) sum_{w=1..8} M_method,w',
                        'regime_weights': {'camera_only': '1/8', 'object_only': '3/8', 'co_motion': '3/8', 'drive_stop_drive': '1/8'},
                        'per_regime': 'arithmetic mean of its windows', 'micro_or_concatenation': False,
                        'empty_windows': 'Retained with zero precision/coverage/F1 and contrast; no difficulty-based exclusions.',
                        'invalid_input': 'Original code errors on missing/invalid input; no silent skip.'},
        'acquisition_history': 'Original fixed AMI caches use 8 warmup intervals with phase initialized at local start. EIRM uses its original continuous causal rollout from frame 0; its state/history is not reset at scored windows. This acquisition-history limitation is disclosed; scoring protocol is unchanged.',
        'eirm_frozen_config': {'checkpoint': str(STUDENT), 'sha256': sha(STUDENT), 'confidence': .5,
                               'require_imu': False, 'dwell_s': .025, 'prism_angle_deg': .5,
                               'contrast_threshold': .2, 'max_phase_step_deg': 30, 'device': 'cpu',
                               'policy': 'original _ours_windows; features at window i use visual/IMU from i-1, previous occupancy, disabled overlap; no oracle at inference',
                               'saved_outputs': 'Save every applied policy action and causal feature vector, and exact scored event stream before scoring; match historical test actions where available.'},
        'environment': {'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__},
        'excluded_results': {'results/ami_next/controlled_adaptive_cm_iwe.json': 'Different 64x64 procedural dataset, windows, prism, checkpoint; not comparable.',
                             'results/ami_next/cm_metrics_summary.json': 'Gallery contrast-only summary; not the source of accepted F1/precision/coverage.'},
    }
    save('protocol_summary.json', protocol)


def validate_frozen():
    p = read(OUT / 'protocol_summary.json')
    assert sha(ORIGINAL) == p['accepted_result']['sha256']
    assert sha(STUDENT) == p['eirm_frozen_config']['sha256']
    for file, digest in p['source_sha256'].items():
        assert sha(file) == digest, file
    for w in p['windows']:
        assert sha(w['rgb_path']) == w['rgb_sha256']
        for source in w['baseline_event_sources'].values():
            assert sha(source['path']) == source['sha256']
    return p


def baseline():
    p = validate_frozen()
    subprocess.run(p['reproduction_command'], check=True)
    old, new = read(ORIGINAL), read(OUT / 'baseline_reproduction.json')
    errors = {}
    for method in old['aggregate']:
        errors[method] = {k: abs(old['aggregate'][method][k] - new['aggregate'][method][k]) for k in KEYS}
    identities = ('split', 'scene', 'regime', 'hero', 'method')
    assert len(old['rows']) == len(new['rows']) == 32
    row_errors = []
    for a, b in zip(old['rows'], new['rows']):
        assert all(a[k] == b[k] for k in identities)
        row_errors.extend(abs(a[k] - b[k]) for k in KEYS + ('event_count', 'cm_event_count'))
    passed = max(row_errors) <= 1e-10 and all(v <= 1e-10 for e in errors.values() for v in e.values())
    save('baseline_gate.json', {'passed': passed, 'tolerance_absolute': 1e-10, 'max_per_window_error': max(row_errors),
                               'aggregate_absolute_errors': errors, 'methods_reproduced': list(errors),
                               'baseline_result_sha256': sha(OUT / 'baseline_reproduction.json')})
    print('BASELINE GATE', passed, errors, flush=True)
    if not passed:
        save('per_regime_cm_iwe_eirm.json', {'status': 'not_evaluated_baseline_gate_failed'})
        save('cm_iwe_all_methods_final.json', {'status': 'not_comparable_baseline_gate_failed'})
        (OUT / 'report.md').write_text('# CM-IWE reproduction stopped\n\nBaseline replay failed the 1e-10 tolerance. Adaptive EIRM was not evaluated. See baseline_gate.json for per-method differences and protocol_summary.json for frozen inputs.\n')
        raise SystemExit('STOP: baseline mismatch; adaptive evaluation prohibited.')


def adaptive(window_indices=None):
    p = validate_frozen()
    gate = read(OUT / 'baseline_gate.json')
    assert gate['passed'], 'Baseline gate must pass before adaptive evaluation'
    assert sha(OUT / 'baseline_reproduction.json') == gate['baseline_result_sha256']
    from ami_simulator.geometry import RotatingWedgePrism
    from ami_simulator.motion_net import TinyCausalGearPolicy
    from ami_simulator.types import EventStream
    from scripts import evaluate_dl3dv_downstream as rollout
    from scripts.evaluate_eveyes_v2_downstream import _imu_fraction, _intrinsics
    checkpoint = torch.load(STUDENT, map_location='cpu', weights_only=True)
    model = TinyCausalGearPolicy(feature_dim=4, hidden_dim=int(checkpoint['hidden_dim']), num_gears=int(checkpoint['num_gears']))
    model.load_state_dict(checkpoint['model'])
    model.eval()
    torch.set_num_threads(8)
    rows = []
    selected_windows = p['windows'] if window_indices is None else [p['windows'][i] for i in window_indices]
    for w in selected_windows:
        scene, h = Path(w['scene_path']), w['hero']
        scene_id = f"{w['split']}/{w['scene']}/{w['regime']}"
        stem = scene_id.replace('/', '_') + f'_f{h}'
        row_path = OUT / 'eirm_windows' / f'{stem}.json'
        if row_path.exists():
            row = read(row_path)
            for file, digest in row['artifacts_sha256'].items():
                assert sha(file) == digest
            rows.append(row)
            continue
        print('EIRM rollout', scene_id, 'through', h, flush=True)
        frames, times = load_rgb_sequence(scene / 'images', scene / 'timestamps.csv', timestamp_unit='s', max_frames=h + 2)
        lum = rgb_to_linear_luminance(frames)
        del frames
        reference = compute_reference_edge_mask(lum[h], edge_quantile=.75)
        assert array_sha(reference) == w['reference_edge_sha256']
        masks = np.ones(lum.shape, dtype=bool)
        visual = rollout._visual_motion(lum, times)
        imu = _imu_fraction(scene / 'imu.csv', times, tuple(checkpoint['imu_motion_bounds']))
        records, pending = [], {}
        original_step, original_choose = rollout._step, rollout._choose_rpm

        def recorded_choose(*args, **kwargs):
            pending['features'] = kwargs['features'][-1].tolist()
            return original_choose(*args, **kwargs)

        def recorded_step(*args, **kwargs):
            index, rpm = args[4], args[5]
            result = original_step(*args, **kwargs)
            records.append({'window': index, 't0': float(times[index]), 't1': float(times[index+1]),
                            'rpm': float(rpm), 'causal_features': pending['features'],
                            'event_count': len(result['reference_events']),
                            'phase_start': float(args[6]), 'phase_end': float(result['phase_end'])})
            # Keep the original rollout state and decision logic; avoid retaining
            # hundreds of millions of events that are outside evaluated windows.
            if index != h:
                result['reference_events'] = EventStream.empty()
            if index % 30 == 0:
                print('  window', index, 'rpm', rpm, flush=True)
            return result

        rollout._choose_rpm, rollout._step = recorded_choose, recorded_step
        try:
            streams, rpms = rollout._ours_windows(lum, times, masks, RotatingWedgePrism(_intrinsics(scene), .5),
                visual=visual, imu=imu, model=model, checkpoint=checkpoint,
                confidence=.5, require_imu=False, dwell=.025, device='cpu')
        finally:
            rollout._choose_rpm, rollout._step = original_choose, original_step
        events = streams[h]
        historical = None
        if w['split'] == 'test':
            source = Path('results/ami_next/eveyes_v2_drive_stop_drive_ours_m0.json' if w['regime'] == 'drive_stop_drive'
                          else 'results/ami_next/eveyes_v2_regimes_test_ours_m0.json')
            previous = sorted([r for r in read(source)['rows'] if r['method'] == 'ours' and r['scene'] == scene_id and r['window'] <= h], key=lambda r:r['window'])
            historical = {'source': str(source), 'sha256': sha(source), 'windows_compared': len(previous),
                          'action_mismatches': [i for i, (a,b) in enumerate(zip(previous,records)) if a['rpm'] != b['rpm']],
                          'hero_event_count_original': previous[-1]['event_count'], 'hero_event_count_replayed': len(events)}
            assert len(previous) == len(records)
            assert not historical['action_mismatches'], historical
        actions_path = OUT / 'policy_outputs' / f'{stem}.json'
        save(str(actions_path.relative_to(OUT)), {'checkpoint_sha256': sha(STUDENT), 'scene': scene_id,
                'source': 'frozen causal policy inference; no oracle outputs used', 'rows': records, 'historical_check': historical})
        event_path = OUT / 'events' / f'{stem}.npz'
        event_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(event_path, x=events.x, y=events.y, t=events.t, p=events.p)
        # Scoring explicitly reloads the saved outputs and saved event stream.
        saved_rpm = read(actions_path)['rows'][h]['rpm']
        with np.load(event_path) as data:
            events = EventStream(data['x'],data['y'],data['t'],data['p'])
        predicted, cm = _cm_edges(events, tuple(w['image_shape_hw']), w['t1'], 80000)
        row = {k:w[k] for k in ('split','scene','regime','hero','t0','t1','reference_edge_sha256','valid_mask_sha256')}
        row.update(method='adaptive_eirm', rpm=saved_rpm, event_count=len(events),
                   policy_rollout_mean_rpm=float(np.mean(rpms)), historical_check=historical,
                   artifacts_sha256={str(actions_path): sha(actions_path), str(event_path): sha(event_path)},
                   **_edge_f1(predicted,reference), **cm)
        save(str(row_path.relative_to(OUT)), row)
        rows.append(row)
        print('EIRM scored', scene_id, {k:row[k] for k in KEYS}, flush=True)
        del lum, masks, streams, events
    if window_indices is not None:
        print('Selected windows completed; run adaptive without indices to validate and aggregate all eight.', flush=True)
        return
    save('eirm_rows.json', rows)
    aggregate = {k:float(np.mean([r[k] for r in rows])) for k in KEYS}
    per_regime = {}
    for regime in (*REQUESTED, 'drive_stop_drive'):
        selected = [r for r in rows if r['regime'] == regime]
        per_regime[regime] = {'n_windows': len(selected), 'mean_rpm': float(np.mean([r['rpm'] for r in selected])) if selected else None,
            **{k:float(np.mean([r[k] for r in selected])) if selected else None for k in KEYS},
            'status': 'evaluated' if selected else 'absent_from_original_protocol'}
    save('per_regime_cm_iwe_eirm.json', {'protocol': p['protocol'], 'scope': p['scope_discrepancy'],
        'per_regime': per_regime, 'mean_rpm_definition': 'mean selected RPM over scored windows only, not entire rollout', 'rows': rows})
    baseline_report = read(OUT / 'baseline_reproduction.json')
    identity = lambda r: tuple(r[k] for k in ('split', 'scene', 'regime', 'hero'))
    expected = [identity(w) for w in p['windows']]
    assert [identity(r) for r in rows] == expected
    for method in baseline_report['aggregate']:
        assert [identity(r) for r in baseline_report['rows'] if r['method'] == method] == expected
    assert all(r['cm_event_count'] == min(r['event_count'],80000) for r in rows + baseline_report['rows'])
    all_aggregate = {m:{k:v[k] for k in KEYS} for m,v in baseline_report['aggregate'].items()}
    all_aggregate['adaptive_eirm'] = aggregate
    checks = {
        '1_baselines_reproduced': gate,
        '2_identical_windows': {'passed': len(rows)==8, 'n_each_method': 8},
        '3_identical_rgb_reference_edges': {'passed': True, 'evidence': 'reference hashes checked per window'},
        '4_identical_cm_grid': {'passed': True, 'count':63, 'scorer':'original imported _cm_edges'},
        '5_identical_valid_region': {'passed':True, 'mask':'full native image; per-window mask hashes'},
        '6_compensation_convention': {'passed':True, 'convention':p['compensation'], 'acquisition_history_caveat':p['acquisition_history']},
        '7_sample_counts': {'passed':True, 'evaluation_windows_each':8, 'cm_input_cap_each':80000, 'note':'Event counts differ by sensing method; same cap rule, no padding sparse streams.'},
        '8_causal_saved_policy_outputs': {'passed':True, 'historical_actions_checked':[r['historical_check'] for r in rows if r['historical_check']], 'new_train_valid_outputs':'same frozen checkpoint; saved causal features/actions; no oracle selection'},
        '9_contrast_diagnostic_only': {'passed':True, 'primary_metric':'CM edge F1'},
        'requested_four_regimes_complete': {'passed':False, 'reason':p['scope_discrepancy']},
    }
    save('cm_iwe_all_methods_final.json', {'protocol':p['protocol'], 'status':'original_eight_window_protocol_matched; requested_counter_motion_unavailable',
        'scope':p['scope_discrepancy'], 'n_windows_each_method':8, 'aggregate':all_aggregate,
        'per_regime':per_regime, 'aggregation':p['aggregation'], 'checks':checks,
        'baseline_rows':baseline_report['rows'], 'eirm_rows':rows})
    print(json.dumps({'aggregate':all_aggregate, 'per_regime':per_regime},indent=2))


def report():
    p = validate_frozen()
    final = read(OUT / 'cm_iwe_all_methods_final.json')
    gate = read(OUT / 'baseline_gate.json')
    labels = {'dvs_voltmeter':'DVS', 'v2e':'v2e', 'ami_fixed_180':'Fixed 180', 'ami_fixed_360':'Fixed 360', 'adaptive_eirm':'Adaptive EIRM',
              'camera_only':'Camera-only','object_only':'Object-only','co_motion':'Co-motion','counter_motion':'Counter-motion','drive_stop_drive':'Drive–stop–drive'}
    lines = ['# CM-IWE protocol recovery and matched EIRM evaluation', '',
        '**Scope correction:** ' + p['scope_discrepancy'], '',
        'The requested four-regime aggregate cannot be produced under the unchanged accepted protocol: Counter-motion has zero original windows. The table below reproduces and extends the original eight-window summary, retaining Drive–stop–drive without relabeling it.', '',
        '## Original protocol', '',
        'Source: `scripts/evaluate_cm_edge_reconstruction.py`; accepted results: `' + str(ORIGINAL) + '`. Exact settings, source hashes, cached event inputs, RGB edge hashes, window timestamps and replay command are in `protocol_summary.json`. A historical literal CLI was not located; the effective invocation is reconstructed and verified by baseline replay.', '',
        '- Native EVEYES 960×540, one-frame windows at 60 Hz. RGB edges use frame h at t0; the CM warp uses t1, preserving the original convention.',
        '- 63 global translation candidates (pixels/s): vx = [-160,-80,-40,-20,0,20,40,80,160], vy = [-80,-40,-20,0,20,40,80]. Nonuniform Cartesian grid; first maximum wins ties.',
        '- Deterministic linspace subsampling to at most 80,000 events per window. Polarity-separated count images select the warp by summed population variance. Polarity counts are then summed for edge extraction.',
        '- Reference edges: gradient magnitude of log linear luminance; predicted edges: gradient magnitude of log1p IWE. Both use their own positive-gradient 75th percentile. Exact pixel overlap, zero-pixel tolerance, full image, no target mask or crop.',
        '- Precision = overlap/predicted edges; Coverage = overlap/reference edges; F1 = harmonic mean. Empty denominators return zero and windows are retained. Missing/invalid inputs cause failure, not silent exclusion.',
        '- AMI uses per-event prism compensation; DVS/v2e have no prism and use the reference plane directly. CM contrast is a diagnostic, not the primary reconstruction metric.', '',
        '| Split | Scene | Regime | Frame | t0 (s) | t1 (s) |', '|---|---|---|---:|---:|---:|']
    for w in p['windows']:
        lines.append(f"| {w['split']} | {w['scene']} | {labels[w['regime']]} | {w['hero']} | {w['t0']:.9f} | {w['t1']:.9f} |")
    lines += ['', '## Baseline reproduction gate', '',
        'All four baselines were re-scored from the frozen original event streams, using the original evaluator. All 32 per-window rows and all four reported metrics pass absolute tolerance 1e-10. Maximum per-window numerical difference: ' + str(gate['max_per_window_error']) + '. This verifies scoring replay; it does not claim regeneration of the external event backends.', '',
        '## EIRM by regime', '', '| Regime | N windows | Mean RPM | F1 | Precision | Coverage | CM contrast |', '|---|---:|---:|---:|---:|---:|---:|']
    for regime, values in final['per_regime'].items():
        numbers = ['—' if values[k] is None else f'{values[k]:.4f}' for k in ('mean_rpm',)+KEYS]
        lines.append('| ' + labels[regime] + ' | ' + str(values['n_windows']) + ' | ' + ' | '.join(numbers) + ' |')
    lines += ['', 'Mean RPM averages only the scored windows. It is not the mean RPM over each complete rollout. Counter-motion is unavailable, not a zero metric.', '',
        '## Original eight-window aggregate', '', '| Method | CM edge F1 | Precision | Coverage | CM contrast |', '|---|---:|---:|---:|---:|']
    for method, values in final['aggregate'].items():
        lines.append('| ' + labels[method] + ' | ' + ' | '.join(f'{values[k]:.4f}' for k in KEYS) + ' |')
    lines += ['', '**Aggregation:** M(method) = (1/8) Σ(window=1…8) M(method, window), applied separately to F1, Precision, Coverage and CM contrast. F1 is averaged after window-level computation. Regime weights are 1/8 Camera-only, 3/8 Object-only, 3/8 Co-motion, 1/8 Drive–stop–drive. This is neither equal-regime macro aggregation nor micro aggregation.', '',
        '## Causal policy and limitations', '',
        'The frozen native EVEYES checkpoint `' + str(STUDENT) + '` is used at confidence 0.5, no IMU gate, dwell 0.025 s, prism 0.5 degrees. It uses lagged visual/IMU and previous aligned occupancy; no oracle action selection or CM-based retuning occurs. Applied actions and causal features are saved before CM scoring in `policy_outputs/`, and scored event streams in `events/`. Historical test action traces are checked against replay; train/valid use the same frozen checkpoint with newly saved causal outputs.', '',
        p['acquisition_history'], '',
        'The original eight windows include train and validation scenes used by the project, and were originally selected as representative gallery windows. These results therefore must not be presented as a new held-out four-regime generalization benchmark. Matching the original scoring protocol does not remove those original limitations.', '',
        'The previous `controlled_adaptive_cm_iwe.json` uses a different 64×64 procedural dataset, window set, prism and checkpoint. Its Fixed180/360 values are not comparable to this table and are excluded. `cm_metrics_summary.json` is a gallery contrast-only diagnostic; it is not the source of the accepted F1 table.', '',
        '## Consistency checks', '']
    for key, value in final['checks'].items():
        lines.append('- ' + key + ': ' + ('PASS' if value.get('passed') else 'NOT AVAILABLE / FAIL') + '.')
    lines += ['', '## Reproduce', '', '```bash', 'python -m scripts.audit_cm_iwe_protocol freeze',
              'python -m scripts.audit_cm_iwe_protocol baseline', 'python -m scripts.audit_cm_iwe_protocol adaptive',
              'python -m scripts.audit_cm_iwe_protocol report', '```', '',
              'Freeze is intentionally refused when a manifest already exists. For existing frozen outputs, run the baseline/adaptive/report stages; adaptive validates and reuses saved windows. Original accepted results are untouched.', '']
    (OUT / 'report.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze','baseline','adaptive','report'))
    parser.add_argument('--window-indices', nargs='+', type=int, help='Independent adaptive workers; does not create aggregate outputs')
    args = parser.parse_args()
    if args.stage == 'adaptive':
        adaptive(args.window_indices)
    else:
        if args.window_indices is not None:
            parser.error('--window-indices is only valid for adaptive')
        {'freeze':freeze, 'baseline':baseline, 'report':report}[args.stage]()
