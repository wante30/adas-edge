"""Exercise native decisions through the real HUD1 codec and software receiver.

All cases are deterministic software fixtures, not video recognition results.
No command line option can select hardware or a physical PCIe transport.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from competition_ai.scenario import Boundary, Controls, Lane, Light, Maneuver, Mode, Motion, Perception
from competition_ai.scenario_policy import decide
from competition_io.scenario_hud import MockScenarioHud, project_scenario


def run(output: Path) -> dict:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    green = Perception(Light.GREEN, True, 0)
    red = Perception(Light.RED, True, 0)
    road_left = Controls(mode=Mode.ROAD, request=Maneuver.CHANGE_LEFT)
    cases = [
        ('green_left_turn', Controls(lane=Lane.LEFT), green, 0, 0, 0x103, 1),
        ('green_straight', Controls(), green, 0, 0, 0x101, 0),
        ('green_right_turn', Controls(lane=Lane.RIGHT), green, 0, 0, 0x105, 2),
        ('red_stop', Controls(), red, 0, 0, 0x101, 0),
        ('red_forced_go', Controls(motion=Motion.GO), red, 0, 0, 0x109, 4),
        ('pedestrian_brake', Controls(), replace(green, pedestrian_hazard=True, hazard_timestamp_ns=0), 0, 0, 0x109, 4),
        ('collision_brake', Controls(), replace(green, obstacle_hazard=True, hazard_timestamp_ns=0), 0, 0, 0x109, 4),
        ('dashed_left_change', road_left, Perception(left_boundary=Boundary.DASHED, lane_timestamp_ns=0), 0, 0, 0x103, 1),
        ('solid_left_rejected', road_left, Perception(left_boundary=Boundary.SOLID, lane_timestamp_ns=0), 0, 0, 0x109, 4),
        ('unknown_lane_rejected', road_left, Perception(), 0, 0, 0x109, 4),
        ('unknown_signal_stop', Controls(), Perception(), 0, 0, 0x109, 4),
        ('missing_navigation_stop', Controls(mode=Mode.NAVIGATION), Perception(), 0, 0, 0x109, 4),
        ('future_source', Controls(lane=Lane.LEFT), green, 0, 1, 0x109, 4),
        ('expired_source', Controls(lane=Lane.LEFT), green, 750_000_000, 0, 0x109, 4),
        ('future_green_observation', Controls(lane=Lane.LEFT), replace(green, signal_timestamp_ns=1), 0, 0, 0x109, 4),
    ]
    sink = MockScenarioHud()
    receipts = []
    for index, (name, controls, perception, now, source, expected_state, expected_blink) in enumerate(cases):
        decision = decide(controls, perception, now)
        projection = project_scenario(controls, perception, decision, now_ns=now,
                                      source_frame_id=index, source_timestamp_ns=source)
        receipt = sink.submit(projection)
        receipt.update(case=name, input_kind='DETERMINISTIC_SOFTWARE_FIXTURE_NOT_VIDEO',
                       expected_state=expected_state, expected_blink=expected_blink)
        receipt['passed'] = (receipt['receiver_verified'] and projection.state == expected_state
                             and projection.blink == expected_blink)
        receipts.append(receipt)
    paths = ['competition_io/scenario_hud.py', 'competition_io/hud_v1.py',
             'competition_io/hud_client.py', 'competition_io/hud_simulator.py',
             'competition_ai/scenario_policy.py', 'tools/audit_scenario_hud.py']
    report = {
        'evidence': 'PC_SOFTWARE_CODEC_CLIENT_REGISTER_RECEIVER_ONLY',
        'passed': all(item['passed'] for item in receipts),
        'case_count': len(receipts), 'passed_cases': sum(item['passed'] for item in receipts),
        'physical_pcie_or_fpga_tested': False, 'hardware_writes': 0,
        'legacy_adst_generated': False,
        'source_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in paths},
        'limitations': ['HUD1 carries icons only: straight, stop, brake, colours and lane semantics are sidecar-only',
                        'TTL/source-time validation occurs before submission; HUD1 has no receiver freshness watchdog',
                        'The simulator confirms software accepted/applied registers, not real PCIe, CDC, HDMI or LED wiring',
                        'No new native maneuver is encoded as legacy ADST LEFT/RIGHT'],
        'cases': receipts,
    }
    (output/'scenario_hud_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(output), 'passed': report['passed'], 'cases': len(receipts)}, ensure_ascii=False))
    if not report['passed']:
        raise AssertionError('A scenario did not match the expected HUD1 projection or receiver state')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New evidence directory; existing output is preserved')
    run(parser.parse_args().output)
