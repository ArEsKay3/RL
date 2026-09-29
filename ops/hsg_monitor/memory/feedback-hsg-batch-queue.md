---
name: feedback-hsg-batch-queue
description: "rkirby 2026-09-29: on HSG there is no reservation; all campaign jobs go to the regular batch queue on account nemotron_sw_post"
metadata:
  type: feedback
---

rkirby, 2026-09-29 (HSG, typed directly): "We have no reservation here so will be working on the regular batch queue on the nemotron_sw_post account."

**Why:** CMH ran on the sla_res_nemotron_sw_post reservation with QOS hero-res and 8 h segments; none of that exists on HSG.

**How to apply:** partition `batch` (4 h max wall, so 4 h segments), QOS `normal` (`short` only for smokes), account `nemotron_sw_post`; do not ask about or look for a reservation. Related: [[hsg-cluster-facts]], [[feedback-rkirby-timeslices-the-block]] (CMH-era, now moot).
