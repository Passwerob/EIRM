Snapshot before starting a separate motion-speed negative-feedback controller.

Do not overwrite:
  results/snapshots/2026-09-10_eveyes_v2_m0/     paper controller m0
  results/ami_next/figures/eveyes_event_qualitative.*  live copies of this freeze

Frozen here:
  figures/eveyes_event_qualitative.{pdf,png,json}
    object_only only; RGB 0/2/6 s; black polarity events
    Idle = DVS; storm = 1080 RPM sensor-plane 3-window exaggeration
    180 = inverse-warped reference
  scripts/plot_eveyes_event_qualitative.py
  picks.json
  policy/policy.pt   m0 (same file as 2026-09-10 snapshot)

Paper Table 1 and m0 stay frozen. New motion-control work goes in a new results/ dir.

Date: 2026-09-12
