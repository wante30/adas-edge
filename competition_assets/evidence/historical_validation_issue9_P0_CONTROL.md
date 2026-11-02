# P0 FPGA control acceptance

Baseline: 46400aa1a3610fe6537e0c2962ff991238e70c0d.

`tools/start_competition.sh` explicitly enables real HARDWARE competition control from physical Q4. `tools/start_pcie.sh` remains the no-send performance entry. Restart the saved competition session with the competition launcher; do not use a performance launcher to resume competition.

Three bounded manual ARM -> FPGA brake commands passed on boot 8161ae7d-7d47-49f2-baa0-ed293d3d117d. See p0_fpga_manual_control.json: request_id, FPGA accepted/active sequence and LVT1 last_sequence/current_action/brake_allow are correlated; each ended. These are manual hardware tests, not AI trigger claims or proof of mechanical braking.

HARDWARE rejects synthetic events and requires explicit fpga_accepted in the ACK. Existing critical/fresh/measured/source/epoch/path eligibility is retained. Seven AutoBrake regression tests passed locally.

Original autostart, services, binaries and PR8 release remain in place. Isolated test_session saves original process/service state; `python3 tools/test_session.py restore` restores it. The original service retains its prior diagnostic authorization override; use the explicit competition launcher for the new authorized runtime.
