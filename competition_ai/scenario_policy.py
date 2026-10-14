"""Competition policy that separates turns, lane changes and simulated motion."""
from __future__ import annotations

from .scenario import Boundary, Controls, Decision, Direction, Lane, Light, Maneuver, Mode, Motion, Perception


def decide(controls: Controls, perception: Perception, now_ns: int) -> Decision:
    if not isinstance(now_ns, int) or now_ns < 0:
        raise ValueError("now_ns must be a nonnegative source-clock timestamp")

    def fresh(timestamp):
        return timestamp is not None and 0 <= now_ns-timestamp < perception.valid_for_ms*1_000_000

    def output(maneuver, allowed, alarms=(), reason="valid"):
        direction = {Maneuver.FORWARD: Direction.FORWARD,
                     Maneuver.TURN_LEFT: Direction.LEFT, Maneuver.CHANGE_LEFT: Direction.LEFT,
                     Maneuver.TURN_RIGHT: Direction.RIGHT, Maneuver.CHANGE_RIGHT: Direction.RIGHT}.get(maneuver, Direction.NONE)
        motion = controls.motion
        if motion == Motion.AUTO:
            motion = Motion.GO if direction != Direction.NONE else Motion.STOP
        hint = controls.navigation if controls.navigation_valid else Direction.NONE
        return Decision(maneuver, direction, allowed, tuple(alarms), reason, motion, hint)

    if perception.pedestrian_hazard or perception.obstacle_hazard:
        if not fresh(perception.hazard_timestamp_ns):
            return output(Maneuver.STOP, False, ("STALE_HAZARD",), "hazard_observation_unconfirmed")
        alarm = "PEDESTRIAN_DANGER" if perception.pedestrian_hazard else "COLLISION_WARNING"
        return output(Maneuver.BRAKE, False, (alarm,), "hazard_priority")

    signal_required = controls.mode == Mode.INTERSECTION or controls.signalized
    if signal_required:
        if not perception.signal_associated:
            return output(Maneuver.STOP, False, ("SIGNAL_UNASSOCIATED",), "selected_lane_signal_unconfirmed")
        if not fresh(perception.signal_timestamp_ns):
            return output(Maneuver.STOP, False, ("STALE_SIGNAL",), "source_signal_expired_or_missing")
        if perception.light == Light.RED:
            alarms = ("RED_LIGHT",) if controls.motion == Motion.GO else ()
            return output(Maneuver.STOP, False, alarms, "red_light_stop")
        if perception.light == Light.YELLOW:
            return output(Maneuver.STOP, False, ("YELLOW_LIGHT",), "yellow_light_stop")
        if perception.light != Light.GREEN:
            return output(Maneuver.STOP, False, ("UNKNOWN_SIGNAL",), "unknown_signal_stop")

    if controls.request == Maneuver.STOP or controls.motion == Motion.STOP:
        return output(Maneuver.STOP, True, reason="manual_stop")
    request = controls.request
    if request == Maneuver.AUTO:
        if controls.mode == Mode.INTERSECTION:
            request = {Lane.LEFT: Maneuver.TURN_LEFT, Lane.STRAIGHT: Maneuver.FORWARD,
                       Lane.RIGHT: Maneuver.TURN_RIGHT}[controls.lane]
        else:
            request = Maneuver.FORWARD
    if request in (Maneuver.CHANGE_LEFT, Maneuver.CHANGE_RIGHT):
        boundary = perception.left_boundary if request == Maneuver.CHANGE_LEFT else perception.right_boundary
        if not fresh(perception.lane_timestamp_ns) or boundary == Boundary.UNKNOWN:
            return output(Maneuver.FORWARD, False, ("UNKNOWN_LANE",), "lane_change_boundary_unconfirmed")
        if boundary != Boundary.DASHED:
            return output(Maneuver.FORWARD, False, ("SOLID_LANE",), "solid_lane_change_forbidden")
        return output(request, True, reason="dashed_lane_change_allowed")
    if request in (Maneuver.TURN_LEFT, Maneuver.TURN_RIGHT):
        if controls.mode not in (Mode.INTERSECTION, Mode.NAVIGATION):
            return output(Maneuver.FORWARD, False, ("WRONG_SCENE",), "turn_request_outside_turn_scene")
        if controls.mode == Mode.INTERSECTION:
            expected = Lane.LEFT if request == Maneuver.TURN_LEFT else Lane.RIGHT
            if controls.lane != expected:
                return output(Maneuver.STOP, False, ("WRONG_LANE",), "turn_does_not_match_selected_lane")
        return output(request, True, reason="turn_allowed")
    # Navigation is a route hint, not an immediate lateral manoeuvre. A turn
    # request/approach event must explicitly activate the turn above.
    return output(Maneuver.FORWARD, True, reason="forward_allowed")
