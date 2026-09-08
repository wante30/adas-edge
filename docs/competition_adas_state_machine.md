# Competition ADAS state machine

`CompetitionAdas` is the deterministic policy boundary between scene metadata and an action. Its priority is:

1. pedestrian danger or obstacle hazard -> `BRAKE` with a warning flag;
2. red traffic light while moving -> `STOP + RED_LIGHT`;
3. yellow or unknown traffic light -> conservative `STOP` with `YELLOW_LIGHT` or `TRAFFIC_LIGHT_STATE_UNKNOWN`;
4. a requested change on a dashed lane -> `LEFT` or `RIGHT`;
5. a requested change on a solid or unknown lane -> `WARNING` with the corresponding safety flag;
6. otherwise -> `GO` (or `STOP` when the vehicle command is already `stop`).

A red light while already stopped returns `STOP` without repeating the red-light alarm. COCO `traffic light` detection does not provide color by itself; the HSV ROI adapter may populate the state, and ambiguous ROIs remain `UNKNOWN`. FPGA lane processing, LED/OSD rendering and physical actuator output remain separate adapters.
