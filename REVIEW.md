# v2.3 review summary (-> v2.4)

Created using Anthropic Claude. Internal draft: keep on internal versions until a human has reviewed
and verified it. Share only with people authorised to see the source material.

Bugs found and fixed: see CHANGES.md. Limits of the evidence:
- Verified with synthetic images/videos and no pretrained weights (offline sandbox). Re-check scores
  after `python scripts/download_model.py`, on your own photos.
- Viewpoint-matched video frame selection is unit-tested; its benefit on real walk-around footage is unmeasured.
- Trust-score thresholds remain uncalibrated: PSNR reacts to lighting/noise, so a clean pair can score FAIR.
  The outlined regions are the more reliable signal. Consider exposure-compensated PSNR.
- Not fixed (documented): refresh tokens are not single-use.
- Privacy: the tool stores consumer photos/names/plates; needs Privacy Legal approval before real data.
  Score-driven customer charges need Technology Intake review. The score is a triage aid, not a decision.
