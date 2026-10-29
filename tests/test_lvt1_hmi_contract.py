"""Static contracts for the LVT1/Qt HMI upgrade.

These checks do not replace a Qt build; they prevent the most common
regressions: losing the LVT1 registration, binding 3D lane state to a
constant, using last_action as the current request, or dropping the
relative-only label.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QT = ROOT / 'deployment/lubancat/qt_gpu_probe'


def read(name: str) -> str:
    return (QT / name).read_text(encoding='utf-8')


def test_vehicle_telemetry_registered_and_optional():
    main = read('main.cpp')
    assert 'vehicle_telemetry_status.h' in main
    assert 'qmlRegisterType<VehicleTelemetryStatus>' in main
    assert 'final : public QObject' not in read('vehicle_telemetry_status.h')


def test_lane_card_and_ego_card_exist():
    qml = read('main.qml')
    assert 'id: laneCard' in qml
    assert 'id: egoCard' in qml
    assert '车道感知' in qml
    assert '本车控制' in qml
    assert 'lvtLaneBadge' in qml
    assert 'lvtActionState' in qml


def test_current_action_not_last_action():
    qml = read('main.qml')
    assert 'egoState.action' in qml
    # last_action is history only and must not drive the active request.
    assert 'last_action' not in qml
    assert 'vehicleTelemetry.egoState.active' in qml


def test_brake_is_not_claimed_as_physical():
    qml = read('main.qml')
    assert '刹车允许≠已制动' in qml
    assert '允许≠物理执行' in qml


def test_tesla_binds_lane_state_and_relative_only():
    qml = read('main.qml')
    assert 'laneState: vehicleTelemetry.laneState' in qml
    assert 'displayView: video.aiDisplayView' in qml
    assert 'roadQuadrant: video.aiRoadQuadrant' in qml
    assert 'RELATIVE_ONLY' in qml
    assert 'SCHEMATIC ROAD / LVT1 STYLE' in qml


def test_scene_supports_lane_state_and_not_demo_by_default():
    header = read('tesla_scene_item.h')
    cpp = read('tesla_scene_item.cpp')
    assert 'laneState' in header
    assert 'displayView' in header
    assert 'roadQuadrant' in header
    assert 'laneEdge' in cpp
    assert 'laneEdge(v, -7.105f, leftType)' in cpp
    assert 'laneEdge(v, 7.105f, rightType)' in cpp
    assert 'projection.scale(-1.0f, -1.0f, 1.0f)' in cpp
    assert 'demoMode: false' in read('main.qml')
    assert 'lane_state_.value' in cpp
    assert '0.9) it = live_.erase' in cpp


def test_road_default_is_q1_not_navigation_q2():
    qml = read('main.qml')
    main = read('main.cpp')
    bridge = read('ai_bridge.py')
    assert 'pgvRoadQuadrant !== "undefined" ? pgvRoadQuadrant : 0' in qml
    assert 'intArg(args, "--road-quadrant", 0)' in main
    assert 'default=0' in bridge


def test_ai_status_carries_display_view():
    bridge = read('ai_bridge.py')
    video = read('video_item.cpp')
    assert 'status["display_view"]' in bridge
    assert 'status["road_quadrant"]' in bridge
    assert 'ai_display_view_' in video


def test_pcie_local_source_switch_is_wired():
    main = read('main.cpp')
    qml = read('main.qml')
    assert 'source_controller.h' in main
    assert 'sourceSwitch' in main
    assert 'sourceSwitch->setVideo(video)' in main
    assert 'sourceSwitch.toggle()' in qml
    assert 'Qt.Key_O' in qml
    assert '切换' in qml
    assert 'run_local_replay.sh' not in qml  # script is a deployment helper
