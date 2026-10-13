"""ManiSkill tasks and their generation defaults.

The 4-dim tasks use pd_ee_delta_pos. The 7-dim tasks (PegInsertionSide, PlugCharger and
LiftPegUpright) use pd_joint_pos, decided 9.25: converting them to
pd_ee_delta_pose kept only 60-73% of the demos (dp-manip docs/0925-smoke.md §五), below
final-plan §2.3's 90% gate. pd_joint_pos is the expert's own control mode, so the
"conversion" replays the actions unchanged. Its actions are absolute joint targets in
radians (8 dims, about 30% of values outside [-1, 1]): trainers must normalize them and
must not clip executed actions to [-1, 1].

Raw counts leave room for replays that do not end in success (measured on wsl, 9.25):
the 4-dim conversions keep ~100%, PegInsertionSide pd_joint_pos 97% (34/35),
PlugCharger pd_joint_pos 80% (16/20), so PlugCharger generates 600 + 80.
``--control-mode`` of generate_task.py overrides this choice.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    env_id: str
    control_mode: str
    raw_train: int  # expert demos generated from seed 0; the export keeps the first usable ones
    raw_val: int    # expert demos generated from seed 4000


TASKS = {
    "pickcube": Task("PickCube-v1", "pd_ee_delta_pos", 440, 55),
    "stackcube": Task("StackCube-v1", "pd_ee_delta_pos", 440, 55),
    "pushcube": Task("PushCube-v1", "pd_ee_delta_pos", 440, 55),
    "pullcube": Task("PullCube-v1", "pd_ee_delta_pos", 440, 55),
    "peginsertionside": Task("PegInsertionSide-v1", "pd_joint_pos", 440, 55),
    "plugcharger": Task("PlugCharger-v1", "pd_joint_pos", 600, 80),
    "placesphere": Task("PlaceSphere-v1", "pd_ee_delta_pos", 440, 55),
    "liftpegupright": Task("LiftPegUpright-v1", "pd_joint_pos", 440, 55),
}

# docs/final-plan.md §2.2: training pool (seeds 0-3999, first 400 usable) and validation
# demos (seeds 4000-4999, first 50 usable). Test and validation rollout seeds never become demos.
TRAIN_START, VAL_START = 0, 4000
EXPORT_TRAIN, EXPORT_VAL = 400, 50
