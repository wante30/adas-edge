"""Explicit synthetic test helpers; never invoked by production inference."""
def crosswalk_pulse_candidate(event):
    if (event.get("kind")!="PEDESTRIAN_CROSSING_PATH" or event.get("severity")!="critical"
        or not event.get("synthetic") or not event.get("evidence","").startswith("MTL_CROSSWALK")):
        raise ValueError("Pulse requires an explicit synthetic measured crossing scenario")
    return dict(event,kind="CROSSWALK_BRAKE_PULSE",event_type="CROSSWALK_BRAKE_PULSE",
        evidence="SYNTHETIC_MTL_CROSSWALK_MEASURED_PERSON_PULSE_TEST",event_id=event["event_id"]+":pulse")
