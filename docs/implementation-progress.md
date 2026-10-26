# SDD ledger — plan: docs/superpowers/plans/2026-10-25-ai-camera-simulator.md

User authorized implementation after confirming N=1 round-robin, detection 18 FPS.

Ruling: Work in the supplied workspace without Git commits/worktree: .git is a
read-only placeholder, `git rev-parse` reports not a repository. Keep this ledger
and artifacts; there is no Git rollback/history until the user initializes Git.

Ruling: Docker uses Python 3.11; support Python 3.10 for local verification too,
because the host has 3.10. No change to container execution or clock semantics.

Pre-flight shared interfaces: 1->2 typed config/frame contracts; 2->3 plugin loop;
3->4 renderer and events; 3->5 gateway/cache; 5->6 provider normalization;
6->7 worker protocol. No contradictory interfaces after the scheduling revision.

Ruling: Use framed stdin/stdout pipes for the local subprocess instead of a Unix
socket/service; the same protocol works over SSH. This preserves process isolation
and persistent model loads while avoiding a second service to manage. Independent
worker service scaling is outside the one-video baseline.

Ruling: Replace the tentative MediaPipe pose person detector with OpenCV Zoo YOLOX
COCO person bbox adapter, verified on a real sample. Costs more real compute but
does not change the ideal video clock or configurable model boundary.

Ruling: Plugin protocol is tick/render/finalize (instead of five separate
callbacks), and requests/results use JSON-serializable dicts at the wire boundary.
This keeps scheduling/selection inside the plugin and makes custom runtime workers
simple; future plugins implement this documented protocol.

Task 1: complete — validation, redaction, unique run paths, snapshots; unit tests green.
Task 2: complete — exact rational schedules, VFR, segment/business date and wall delay
invariance tested. Fixed encoder codec-context time base after a VFR RED test.
Task 3: complete — mock, IoU tracker, 1-face round-robin, gallery/confirmation,
attendance and blacklist; unit/integration tests green.
Task 4: implemented, Docker acceptance in progress; native MP4 decode acceptance green.
Task 5: complete for baseline strict replay and real fill-miss gateway; native
real/mock replay produces identical events. Additional failure-path checks ongoing.
Task 6: complete on native CPU — three checksummed model artifacts downloaded,
enrolled astronaut_sample, real person/face/embedding -> attendance -> exact replay.
Task 7: implementation exists — SSH persistent Docker worker, SFTP/hash provision,
runtime capability checks and explicit fallback. Hardware acceptance pending active
board session and GPU/NPU; no fake hardware success claim.

Verification: SIM_REAL_CPU=1 .venv/bin/pytest -q -> 34 passed (includes real inference).
Docker daemon works with escalation; buildx absent, legacy builder used.
