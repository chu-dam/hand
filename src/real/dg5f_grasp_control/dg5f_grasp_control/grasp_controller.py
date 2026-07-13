from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional

import numpy as np

from dg5f_grasp_control.config import RuntimeConfig
from dg5f_grasp_control.control_utils import pose_pd
from dg5f_grasp_control.grasp_policy import GraspPolicy
from dg5f_grasp_control.hand_model import (
    FINGER_JOINT_INDEX,
    GRASP_TAU_SIGN,
    JOINT_COUNT,
    selected_fingers,
)
from dg5f_grasp_control.kinematics import tip_jacobian, tip_position
from dg5f_grasp_control.poses import (
    HAND_NORMAL_POSE,
    HAND_PRE_GRASP_POSE,
    POSE_TYPE_TARGETS,
)

FINGER_SWITCH_VIA_THREE_DELAY = 0.5
ADD_FINGER_PRE_GRASP_DELAY = 0.15
ADD_FINGER_PRE_GRASP_KP_SCALE = 1.4
ADD_FINGER_PRE_GRASP_KD_SCALE = 1.2
ADD_FINGER_BLEND_TIME = 0.7

ENVELOP_FINGER_ORDER = [2, 3, 4, 5]
ENVELOP_FINGER_TORQUE_LOCAL_JOINTS = [1, 2, 3]
ENVELOP_THUMB_TORQUE_LOCAL_JOINTS = [2, 3]

PINKY_SPECIAL_COMMAND = 7
PINKY_SPECIAL_GRASP_COUNT = 4
PINKY_FINGER_ID = 5
PINKY_SPECIAL_FIXED_LOCAL_TARGETS = np.array([0.0, -np.pi / 4.0], dtype=np.float64)
G7_THUMB_TRANSITION_FINGER_ID = 1
G7_INDEX_TRANSITION_FINGER_ID = 2
G7_MIDDLE_TRANSITION_FINGER_ID = 3
G7_RING_TRANSITION_FINGER_ID = 4
G7_TRANSITION_FINGER_ID = G7_INDEX_TRANSITION_FINGER_ID
G7_BASE_GRASP_FINGERS = [1, 2, 3, 4]
G7_TRANSITION_FINGER_NAMES = {
    G7_THUMB_TRANSITION_FINGER_ID: "thumb",
    G7_INDEX_TRANSITION_FINGER_ID: "index",
    G7_MIDDLE_TRANSITION_FINGER_ID: "middle",
    G7_RING_TRANSITION_FINGER_ID: "ring",
}


@dataclass
class ControlOutput:
    tau: np.ndarray
    state: str
    state_elapsed: float
    err: np.ndarray = field(default_factory=lambda: np.zeros(JOINT_COUNT, dtype=np.float64))
    grasp_tau: np.ndarray = field(default_factory=lambda: np.zeros(JOINT_COUNT, dtype=np.float64))
    inactive_pd: np.ndarray = field(default_factory=lambda: np.zeros(JOINT_COUNT, dtype=np.float64))
    alpha: Dict[int, float] = field(default_factory=dict)
    cg: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    cv: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    rotation_enabled: bool = False
    use_fingers: List[int] = field(default_factory=list)
    active_finger_count: int = 0
    inactive_pd_target: np.ndarray = field(
        default_factory=lambda: np.full(JOINT_COUNT, np.nan, dtype=np.float64)
    )
    envelop_info: Dict[str, object] = field(default_factory=dict)
    g7_phase: str = "idle"


class GraspController:
    """Hardware-independent DG5F grasp state machine and torque controller.

    This class owns the grasp/pose state, finger switching, groped-grasp
    policy, inactive-finger PD, envelop grasp, and grasp_type=7 manipulation.
    It intentionally does not know about ROS, MuJoCo data structures, gravity
    compensation, friction compensation, or effort publishers.
    """

    def __init__(
        self,
        cfg: RuntimeConfig,
        log: Optional[Callable[[str], None]] = print,
    ):
        self.cfg = cfg
        self._log_fn = log
        self.hand_q = np.zeros(JOINT_COUNT, dtype=np.float64)

        self.pose_type = 1
        self.pre_grasp_pose_type = 2
        self.state = "NORMAL_POSE"
        self.state_start = 0.0
        self.active_finger_count = 0

        self.deferred_finger_count = None
        self.deferred_finger_count_at = None
        self.adding_finger_target_count = None
        self.adding_pre_grasp_fingers = []
        self.adding_ready_at = None
        self.skip_add_pre_grasp_once = False
        self.blending_fingers = []
        self.blend_started_at = None
        self.blend_until = None

        self.inactive_pd_target = np.full(JOINT_COUNT, np.nan, dtype=np.float64)

        self.envelop_hold_pose = None
        self.envelop_started_at = None
        self.envelop_thumb_enabled = False
        self.envelop_thumb_start_at = None
        self.envelop_last_joint_stall_since = None
        self.envelop_last_info = {}

        self.pinky_special_hold_pose = None
        self.grasp_type7_phase = "idle"
        self.grasp_type7_stable_since = None
        self.grasp_type7_rotation_started_at = None
        self.grasp_type7_done_since = None
        self.grasp_type7_rotation_done = False
        self.grasp_type7_last_qdot_max = 0.0
        self.grasp_type7_transition_finger_id = None
        self.grasp_type7_transition_pd_target = None
        self.grasp_type7_transition_pd_err_max = 0.0
        self.grasp_type7_transition_attach_tau_max = 0.0
        self.grasp_type7_transition_attach_started_at = None
        self.grasp_type7_transition_attach_done_since = None
        self.grasp_type7_index_pd_target = None
        self.grasp_type7_index_pd_err_max = 0.0
        self.grasp_type7_index_attach_tau_max = 0.0
        self.grasp_type7_middle_pd_err_max = 0.0
        self.grasp_type7_middle_attach_tau_max = 0.0
        self.grasp_type7_thumb_pd_err_max = 0.0
        self.grasp_type7_thumb_attach_tau_max = 0.0
        self.grasp_type7_ring_pd_err_max = 0.0
        self.grasp_type7_ring_attach_tau_max = 0.0
        self.grasp_type7_cycle_count = 0
        self.grasp_type7_rotation_cg_ref = None

        initial_count = cfg.use_finger_count if 1 <= cfg.use_finger_count <= 5 else 1
        self.use_fingers = selected_fingers(initial_count)
        self.policy = GraspPolicy(self.use_fingers, cfg)

    def _log(self, message: str) -> None:
        if self._log_fn is not None:
            self._log_fn(message)

    def update_config(self, cfg: RuntimeConfig) -> None:
        self.cfg = cfg
        self.policy = GraspPolicy(self.use_fingers, cfg)

    def set_alpha1(self, alpha1: float) -> None:
        if alpha1 < 0.0 or not np.isfinite(alpha1):
            raise ValueError("alpha1 must be finite and >= 0")
        self.update_config(replace(self.cfg, alpha1=float(alpha1)))

    def sync_joint_state(self, q: np.ndarray) -> None:
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (JOINT_COUNT,):
            raise ValueError(f"q must have shape ({JOINT_COUNT},)")
        self.hand_q = q.copy()

    def _current_pre_grasp_pose(self):
        return POSE_TYPE_TARGETS.get(self.pre_grasp_pose_type, HAND_PRE_GRASP_POSE)

    def _pose_target_for_type(self, pose_type):
        return POSE_TYPE_TARGETS.get(pose_type, HAND_NORMAL_POSE)

    def _apply_pose_type_command(self, pose_type, now):
        self.active_finger_count = 0
        self._clear_transition_state()
        self._reset_envelop_grasp()
        self._reset_pinky_special_grasp()

        self.pose_type = pose_type
        if pose_type == 1:
            self.pre_grasp_pose_type = 2
            state = "NORMAL_POSE"
            label = "normal pose"
        else:
            self.pre_grasp_pose_type = pose_type
            state = "PRE_GRASP_POSE"
            label = "default pre-grasp" if pose_type == 2 else "compact pre-grasp"

        self._log(f"[POSE_TYPE] pose_type={pose_type} -> {label}")
        return state, now, 0.0

    def _inactive_pre_grasp_pd(self, active_fingers, qdot, pre_grasp_fingers=None):
        inactive_pd = np.zeros(JOINT_COUNT, dtype=np.float64)
        self.inactive_pd_target[:] = np.nan
        active_set = set(active_fingers)
        pre_grasp_set = set(pre_grasp_fingers or [])

        for finger, idxs in FINGER_JOINT_INDEX.items():
            if finger in active_set:
                continue

            idxs = np.asarray(idxs, dtype=int)
            if (
                finger == PINKY_FINGER_ID
                and self.pinky_special_hold_pose is not None
            ):
                target = self.pinky_special_hold_pose
                kp = self.cfg.pose_kp
                kd = self.cfg.pose_kd
            elif finger in pre_grasp_set:
                target = self._current_pre_grasp_pose()[idxs]
                kp = self.cfg.pose_kp * ADD_FINGER_PRE_GRASP_KP_SCALE
                kd = self.cfg.pose_kd * ADD_FINGER_PRE_GRASP_KD_SCALE
            else:
                target = np.zeros(len(idxs), dtype=np.float64)
                target[0] = HAND_PRE_GRASP_POSE[idxs[0]]
                if finger == PINKY_FINGER_ID:
                    target[1] = HAND_PRE_GRASP_POSE[idxs[1]]
                kp = self.cfg.pose_kp
                kd = self.cfg.pose_kd

            # Save the exact target that is actually sent to the inactive-finger
            # PD controller. This is used only by the periodic debug log.
            self.inactive_pd_target[idxs] = target

            pd, _ = pose_pd(
                target,
                self.hand_q[idxs],
                qdot[idxs],
                kp=kp,
                kd=kd,
                limit=self.cfg.pose_pd_limit,
            )
            inactive_pd[idxs] = pd

        return inactive_pd

    def _clear_transition_state(self):
        self.deferred_finger_count = None
        self.deferred_finger_count_at = None
        self.adding_finger_target_count = None
        self.adding_pre_grasp_fingers = []
        self.adding_ready_at = None
        self.skip_add_pre_grasp_once = False
        self.blending_fingers = []
        self.blend_started_at = None
        self.blend_until = None

    def _reset_envelop_grasp(self):
        self.envelop_hold_pose = None
        self.envelop_started_at = None
        self.envelop_thumb_enabled = False
        self.envelop_thumb_start_at = None
        self.envelop_last_joint_stall_since = None
        self.envelop_last_info = {}

    def _reset_grasp_type7_rotation_state(self):
        self.grasp_type7_phase = "idle"
        self.grasp_type7_stable_since = None
        self.grasp_type7_rotation_started_at = None
        self.grasp_type7_done_since = None
        self.grasp_type7_rotation_done = False
        self.grasp_type7_last_qdot_max = 0.0
        self.grasp_type7_transition_finger_id = None
        self.grasp_type7_transition_pd_target = None
        self.grasp_type7_transition_pd_err_max = 0.0
        self.grasp_type7_transition_attach_tau_max = 0.0
        self.grasp_type7_transition_attach_started_at = None
        self.grasp_type7_transition_attach_done_since = None
        self.grasp_type7_index_pd_target = None
        self.grasp_type7_index_pd_err_max = 0.0
        self.grasp_type7_index_attach_tau_max = 0.0
        self.grasp_type7_middle_pd_err_max = 0.0
        self.grasp_type7_middle_attach_tau_max = 0.0
        self.grasp_type7_thumb_pd_err_max = 0.0
        self.grasp_type7_thumb_attach_tau_max = 0.0
        self.grasp_type7_ring_pd_err_max = 0.0
        self.grasp_type7_ring_attach_tau_max = 0.0
        self.grasp_type7_cycle_count = 0
        self.grasp_type7_rotation_cg_ref = None

    def _start_grasp_type7_rotation_state(self):
        self.grasp_type7_phase = "grasp_stabilizing"
        self.grasp_type7_stable_since = None
        self.grasp_type7_rotation_started_at = None
        self.grasp_type7_done_since = None
        self.grasp_type7_rotation_done = False
        self.grasp_type7_last_qdot_max = 0.0
        self.grasp_type7_transition_finger_id = None
        self.grasp_type7_transition_pd_target = None
        self.grasp_type7_transition_pd_err_max = 0.0
        self.grasp_type7_transition_attach_tau_max = 0.0
        self.grasp_type7_transition_attach_started_at = None
        self.grasp_type7_transition_attach_done_since = None
        self.grasp_type7_index_pd_target = None
        self.grasp_type7_index_pd_err_max = 0.0
        self.grasp_type7_index_attach_tau_max = 0.0
        self.grasp_type7_middle_pd_err_max = 0.0
        self.grasp_type7_middle_attach_tau_max = 0.0
        self.grasp_type7_thumb_pd_err_max = 0.0
        self.grasp_type7_thumb_attach_tau_max = 0.0
        self.grasp_type7_ring_pd_err_max = 0.0
        self.grasp_type7_ring_attach_tau_max = 0.0
        self.grasp_type7_cycle_count = 0
        self.grasp_type7_rotation_cg_ref = None

    def _active_finger_joint_indices(self):
        idxs = []
        for finger in self.use_fingers:
            idxs.extend(FINGER_JOINT_INDEX[finger])
        return np.asarray(idxs, dtype=int)

    def _active_finger_qdot_max(self, qdot):
        idxs = self._active_finger_joint_indices()
        if idxs.size == 0:
            return 0.0
        return float(np.max(np.abs(qdot[idxs])))

    def _set_grasp_type7_active_fingers(self, fingers):
        self.use_fingers = list(fingers)
        self.policy = GraspPolicy(self.use_fingers, self.cfg)

    def _grasp_type7_transition_name(self, finger):
        return G7_TRANSITION_FINGER_NAMES.get(int(finger), f"finger{int(finger)}")

    def _grasp_type7_joint1_phase_name(self, finger):
        if int(finger) == G7_INDEX_TRANSITION_FINGER_ID:
            return "index_joint1_to_45"
        if int(finger) == G7_MIDDLE_TRANSITION_FINGER_ID:
            return "middle_joint1_to_30"
        if int(finger) == G7_THUMB_TRANSITION_FINGER_ID:
            return "thumb_pose_0_140_0_0"
        if int(finger) == G7_RING_TRANSITION_FINGER_ID:
            return "ring_joint1_to_9"
        return f"finger{int(finger)}_joint1_to_target"

    def _grasp_type7_attach_phase_name(self, finger):
        name = self._grasp_type7_transition_name(finger)
        return f"{name}_attach_to_centroid"

    def _grasp_type7_joint1_target_rad(self, finger):
        if int(finger) == G7_INDEX_TRANSITION_FINGER_ID:
            return float(self.cfg.grasp_type7_index_first_joint_target_rad)
        if int(finger) == G7_MIDDLE_TRANSITION_FINGER_ID:
            return float(self.cfg.grasp_type7_middle_first_joint_target_rad)
        if int(finger) == G7_THUMB_TRANSITION_FINGER_ID:
            return float(self.cfg.grasp_type7_thumb_joint1_target_rad)
        if int(finger) == G7_RING_TRANSITION_FINGER_ID:
            return float(self.cfg.grasp_type7_ring_first_joint_target_rad)
        raise ValueError(f"Unsupported grasp_type7 transition finger: {finger}")

    def _grasp_type7_attach_force(self, finger):
        if int(finger) == G7_INDEX_TRANSITION_FINGER_ID:
            return float(self.cfg.grasp_type7_index_attach_force)
        if int(finger) == G7_MIDDLE_TRANSITION_FINGER_ID:
            return float(self.cfg.grasp_type7_middle_attach_force)
        if int(finger) == G7_THUMB_TRANSITION_FINGER_ID:
            return float(self.cfg.grasp_type7_thumb_attach_force)
        if int(finger) == G7_RING_TRANSITION_FINGER_ID:
            return float(self.cfg.grasp_type7_ring_attach_force)
        return float(self.cfg.grasp_type7_index_attach_force)

    def _grasp_type7_attach_tau_limit(self, finger):
        if int(finger) == G7_INDEX_TRANSITION_FINGER_ID:
            return abs(float(self.cfg.grasp_type7_index_attach_tau_limit))
        if int(finger) == G7_MIDDLE_TRANSITION_FINGER_ID:
            return abs(float(self.cfg.grasp_type7_middle_attach_tau_limit))
        if int(finger) == G7_THUMB_TRANSITION_FINGER_ID:
            return abs(float(self.cfg.grasp_type7_thumb_attach_tau_limit))
        if int(finger) == G7_RING_TRANSITION_FINGER_ID:
            return abs(float(self.cfg.grasp_type7_ring_attach_tau_limit))
        return abs(float(self.cfg.grasp_type7_index_attach_tau_limit))

    def _grasp_type7_transition_target(self, finger):
        finger = int(finger)
        idxs = np.asarray(FINGER_JOINT_INDEX[finger], dtype=int)
        if finger == G7_THUMB_TRANSITION_FINGER_ID:
            return np.array([
                float(self.cfg.grasp_type7_thumb_joint1_target_rad),
                float(self.cfg.grasp_type7_thumb_joint2_target_rad),
                float(self.cfg.grasp_type7_thumb_joint3_target_rad),
                float(self.cfg.grasp_type7_thumb_joint4_target_rad),
            ], dtype=np.float64)

        target = POSE_TYPE_TARGETS[2][idxs].copy()
        target[0] = self._grasp_type7_joint1_target_rad(finger)
        return target

    def _start_grasp_type7_finger_joint1_transition(self, finger):
        # Detachment scenario, first step: remove the transition finger from the
        # active grasp set before moving it. The remaining contact spots define
        # the new centroid and force distribution.
        finger = int(finger)
        remaining = [f for f in G7_BASE_GRASP_FINGERS if f != finger]
        self._set_grasp_type7_active_fingers(remaining)

        target = self._grasp_type7_transition_target(finger)
        self.grasp_type7_transition_finger_id = finger
        self.grasp_type7_transition_pd_target = target
        self.grasp_type7_transition_pd_err_max = float("inf")
        self.grasp_type7_transition_attach_tau_max = 0.0
        self.grasp_type7_transition_attach_started_at = None
        self.grasp_type7_transition_attach_done_since = None
        self.grasp_type7_phase = self._grasp_type7_joint1_phase_name(finger)

        name = self._grasp_type7_transition_name(finger)
        self.grasp_type7_index_pd_target = target if finger == G7_INDEX_TRANSITION_FINGER_ID else None
        if finger == G7_INDEX_TRANSITION_FINGER_ID:
            self.grasp_type7_index_pd_err_max = float("inf")
            self.grasp_type7_index_attach_tau_max = 0.0
        if finger == G7_MIDDLE_TRANSITION_FINGER_ID:
            self.grasp_type7_middle_pd_err_max = float("inf")
            self.grasp_type7_middle_attach_tau_max = 0.0
        if finger == G7_THUMB_TRANSITION_FINGER_ID:
            self.grasp_type7_thumb_pd_err_max = float("inf")
            self.grasp_type7_thumb_attach_tau_max = 0.0
        if finger == G7_RING_TRANSITION_FINGER_ID:
            self.grasp_type7_ring_pd_err_max = float("inf")
            self.grasp_type7_ring_attach_tau_max = 0.0

        self._log(
            f"[GRASP_TYPE7] start {name}_joint1_transition "
            f"active_fingers={self.use_fingers}, "
            f"{name}_target_rad={np.round(target, 4).tolist()}, "
            f"joint1_target_rad={target[0]:.4f}, "
            f"tol_rad={self.cfg.grasp_type7_index_pd_tolerance_rad:.4f}"
        )

    def _start_grasp_type7_index_detach_pregrasp(self):
        self._start_grasp_type7_finger_joint1_transition(G7_INDEX_TRANSITION_FINGER_ID)

    def _start_grasp_type7_middle_detach_pregrasp(self):
        self._start_grasp_type7_finger_joint1_transition(G7_MIDDLE_TRANSITION_FINGER_ID)

    def _start_grasp_type7_thumb_detach_pregrasp(self):
        self._start_grasp_type7_finger_joint1_transition(G7_THUMB_TRANSITION_FINGER_ID)

    def _start_grasp_type7_ring_detach_pregrasp(self):
        self._start_grasp_type7_finger_joint1_transition(G7_RING_TRANSITION_FINGER_ID)

    def _start_grasp_type7_transition_attach_to_centroid(self, finger):
        finger = int(finger)
        self.grasp_type7_transition_finger_id = finger
        self.grasp_type7_transition_pd_target = None
        self.grasp_type7_transition_pd_err_max = 0.0
        self.grasp_type7_transition_attach_tau_max = 0.0
        self.grasp_type7_transition_attach_started_at = None
        self.grasp_type7_transition_attach_done_since = None
        self.grasp_type7_phase = self._grasp_type7_attach_phase_name(finger)

        if finger == G7_INDEX_TRANSITION_FINGER_ID:
            self.grasp_type7_index_pd_target = None
            self.grasp_type7_index_pd_err_max = 0.0
            self.grasp_type7_index_attach_tau_max = 0.0
        if finger == G7_MIDDLE_TRANSITION_FINGER_ID:
            self.grasp_type7_middle_pd_err_max = 0.0
            self.grasp_type7_middle_attach_tau_max = 0.0
        if finger == G7_THUMB_TRANSITION_FINGER_ID:
            self.grasp_type7_thumb_pd_err_max = 0.0
            self.grasp_type7_thumb_attach_tau_max = 0.0
        if finger == G7_RING_TRANSITION_FINGER_ID:
            self.grasp_type7_ring_pd_err_max = 0.0
            self.grasp_type7_ring_attach_tau_max = 0.0

        name = self._grasp_type7_transition_name(finger)
        self._log(
            f"[GRASP_TYPE7] {name}_joint1_transition done -> {name}_attach_to_centroid "
            f"active_fingers={self.use_fingers}, "
            f"attach_force={self._grasp_type7_attach_force(finger):.4f}, "
            f"attach_tau_limit={self._grasp_type7_attach_tau_limit(finger):.4f}"
        )

    def _finish_grasp_type7_transition_cycle(self):
        self._set_grasp_type7_active_fingers(G7_BASE_GRASP_FINGERS)
        self.grasp_type7_transition_finger_id = None
        self.grasp_type7_transition_pd_target = None
        self.grasp_type7_transition_pd_err_max = 0.0
        self.grasp_type7_transition_attach_tau_max = 0.0
        self.grasp_type7_transition_attach_started_at = None
        self.grasp_type7_transition_attach_done_since = None
        self.grasp_type7_index_pd_target = None
        self.grasp_type7_index_pd_err_max = 0.0
        self.grasp_type7_index_attach_tau_max = 0.0
        self.grasp_type7_middle_pd_err_max = 0.0
        self.grasp_type7_middle_attach_tau_max = 0.0
        self.grasp_type7_thumb_pd_err_max = 0.0
        self.grasp_type7_thumb_attach_tau_max = 0.0
        self.grasp_type7_ring_pd_err_max = 0.0
        self.grasp_type7_ring_attach_tau_max = 0.0
        self.grasp_type7_cycle_count += 1

        if bool(self.cfg.grasp_type7_repeat_transition_cycle):
            # One full index->middle->thumb->ring relocation cycle is complete.
            # Re-enter the same rotation pipeline with the same rotation_theta_rad
            # sign, so the object keeps rotating in the original direction.
            self.grasp_type7_phase = "grasp_stabilizing"
            self.grasp_type7_stable_since = None
            self.grasp_type7_rotation_started_at = None
            self.grasp_type7_done_since = None
            self.grasp_type7_rotation_done = False
            self.grasp_type7_rotation_cg_ref = None
            self._log(
                "[GRASP_TYPE7] transition cycle done -> repeat rotation "
                f"cycle={self.grasp_type7_cycle_count}, "
                f"active_fingers={self.use_fingers}, "
                f"theta_rad={self.cfg.rotation_theta_rad:.6f}"
            )
            return

        self.grasp_type7_phase = "transition_done"
        self._log(
            "[GRASP_TYPE7] transition cycle done -> transition_done "
            f"cycle={self.grasp_type7_cycle_count}, "
            f"active_fingers={self.use_fingers}"
        )

    def _is_grasp_type7_transition_phase(self):
        return self.grasp_type7_phase in (
            "index_detach_pregrasp",
            "index_joint1_to_45",
            "index_attach_to_centroid",
            "middle_joint1_to_30",
            "middle_attach_to_centroid",
            "thumb_pose_0_140_0_0",
            "thumb_attach_to_centroid",
            "ring_joint1_to_9",
            "ring_attach_to_centroid",
        )

    def _is_grasp_type7_joint1_pd_phase(self):
        return self.grasp_type7_phase in (
            "index_detach_pregrasp",
            "index_joint1_to_45",
            "middle_joint1_to_30",
            "thumb_pose_0_140_0_0",
            "ring_joint1_to_9",
        )

    def _is_grasp_type7_attach_phase(self):
        return self.grasp_type7_phase in (
            "index_attach_to_centroid",
            "middle_attach_to_centroid",
            "thumb_attach_to_centroid",
            "ring_attach_to_centroid",
        )

    def _transition_finger_qdot_max(self, qdot, finger):
        idxs = np.asarray(FINGER_JOINT_INDEX[int(finger)], dtype=int)
        if idxs.size == 0:
            return 0.0
        return float(np.max(np.abs(qdot[idxs])))

    def _calc_grasp_type7_transition_pd(self, qdot):
        pd = np.zeros(JOINT_COUNT, dtype=np.float64)
        if not self._is_grasp_type7_joint1_pd_phase():
            self.grasp_type7_transition_pd_err_max = 0.0
            return pd

        finger = self.grasp_type7_transition_finger_id
        if finger is None or self.grasp_type7_transition_pd_target is None:
            return pd

        idxs = np.asarray(FINGER_JOINT_INDEX[int(finger)], dtype=int)
        pd_local, err = pose_pd(
            self.grasp_type7_transition_pd_target,
            self.hand_q[idxs],
            qdot[idxs],
            kp=self.cfg.pose_kp,
            kd=self.cfg.pose_kd,
            limit=self.cfg.pose_pd_limit,
        )
        pd[idxs] = pd_local
        if int(finger) == G7_THUMB_TRANSITION_FINGER_ID:
            err_max = float(np.max(np.abs(err))) if err.size else 0.0
        else:
            err_max = float(abs(err[0])) if err.size else 0.0
        self.grasp_type7_transition_pd_err_max = err_max

        if int(finger) == G7_INDEX_TRANSITION_FINGER_ID:
            self.grasp_type7_index_pd_err_max = err_max
        elif int(finger) == G7_MIDDLE_TRANSITION_FINGER_ID:
            self.grasp_type7_middle_pd_err_max = err_max
        elif int(finger) == G7_THUMB_TRANSITION_FINGER_ID:
            self.grasp_type7_thumb_pd_err_max = err_max
        elif int(finger) == G7_RING_TRANSITION_FINGER_ID:
            self.grasp_type7_ring_pd_err_max = err_max

        if err_max <= float(self.cfg.grasp_type7_index_pd_tolerance_rad):
            self._start_grasp_type7_transition_attach_to_centroid(finger)

        return pd

    def _grasp_type7_thumb_attach_centroid(self):
        contact_fingers = [2, 3, 4]
        tip_pos = {finger: tip_position(self.hand_q, finger) for finger in contact_fingers}
        points = np.array([tip_pos[finger] for finger in contact_fingers], dtype=np.float64)
        cg_contact = np.mean(points, axis=0)
        thumb_pos = tip_position(self.hand_q, G7_THUMB_TRANSITION_FINGER_ID)
        return cg_contact + float(self.cfg.thumb_centroid_bias) * (thumb_pos - cg_contact)

    def _calc_grasp_type7_transition_attach_tau(self, cv):
        tau = np.zeros(JOINT_COUNT, dtype=np.float64)
        self.grasp_type7_transition_attach_tau_max = 0.0
        if not self._is_grasp_type7_attach_phase():
            return tau

        finger = self.grasp_type7_transition_finger_id
        if finger is None:
            return tau

        finger = int(finger)
        idxs = np.asarray(FINGER_JOINT_INDEX[finger], dtype=int)
        pos = tip_position(self.hand_q, finger)
        target_cv = self._grasp_type7_thumb_attach_centroid() if finger == G7_THUMB_TRANSITION_FINGER_ID else cv
        diff = target_cv - pos
        dist = np.linalg.norm(diff)
        if dist < 1e-9:
            return tau

        fhat = self.cfg.groped_force_direction_sign * diff / dist
        force = self._grasp_type7_attach_force(finger) * fhat
        J = tip_jacobian(self.hand_q, finger, eps=self.cfg.jacobian_eps)
        tau_local = J.T @ force
        tau_local = tau_local * GRASP_TAU_SIGN[idxs]
        tau_limit = self._grasp_type7_attach_tau_limit(finger)
        tau_local = np.clip(tau_local, -tau_limit, tau_limit)
        tau[idxs] = tau_local
        tau_max = float(np.max(np.abs(tau_local)))
        self.grasp_type7_transition_attach_tau_max = tau_max

        if finger == G7_INDEX_TRANSITION_FINGER_ID:
            self.grasp_type7_index_attach_tau_max = tau_max
        elif finger == G7_MIDDLE_TRANSITION_FINGER_ID:
            self.grasp_type7_middle_attach_tau_max = tau_max
        elif finger == G7_THUMB_TRANSITION_FINGER_ID:
            self.grasp_type7_thumb_attach_tau_max = tau_max
        elif finger == G7_RING_TRANSITION_FINGER_ID:
            self.grasp_type7_ring_attach_tau_max = tau_max
        return tau

    def _update_grasp_type7_attach_state(self, qdot, now):
        finger = self.grasp_type7_transition_finger_id
        if finger is None:
            return False

        finger = int(finger)
        qdot_max = self._transition_finger_qdot_max(qdot, finger)
        self.grasp_type7_last_qdot_max = qdot_max

        if self.grasp_type7_transition_attach_started_at is None:
            self.grasp_type7_transition_attach_started_at = now
            self.grasp_type7_transition_attach_done_since = None
            return False

        attach_elapsed = now - self.grasp_type7_transition_attach_started_at
        if attach_elapsed < float(self.cfg.grasp_type7_transition_attach_min_sec):
            self.grasp_type7_transition_attach_done_since = None
            return False

        if qdot_max <= float(self.cfg.grasp_type7_transition_attach_qdot_threshold):
            if self.grasp_type7_transition_attach_done_since is None:
                self.grasp_type7_transition_attach_done_since = now
            elif now - self.grasp_type7_transition_attach_done_since >= float(self.cfg.grasp_type7_transition_attach_hold_sec):
                name = self._grasp_type7_transition_name(finger)
                self._log(
                    f"[GRASP_TYPE7] {name}_attach_to_centroid done "
                    f"qdot_max={qdot_max:.4f}, "
                    f"hold={now - self.grasp_type7_transition_attach_done_since:.2f}s, "
                    f"attach_elapsed={attach_elapsed:.2f}s"
                )
                if (
                    finger == G7_INDEX_TRANSITION_FINGER_ID
                    and bool(self.cfg.grasp_type7_middle_transition_enable)
                ):
                    self._start_grasp_type7_middle_detach_pregrasp()
                elif (
                    finger == G7_MIDDLE_TRANSITION_FINGER_ID
                    and bool(self.cfg.grasp_type7_thumb_transition_enable)
                ):
                    self._start_grasp_type7_thumb_detach_pregrasp()
                elif (
                    finger == G7_THUMB_TRANSITION_FINGER_ID
                    and bool(self.cfg.grasp_type7_ring_transition_enable)
                ):
                    self._start_grasp_type7_ring_detach_pregrasp()
                else:
                    self._finish_grasp_type7_transition_cycle()
        else:
            self.grasp_type7_transition_attach_done_since = None

        return False

    def _update_grasp_type7_rotation_state(self, qdot, now):
        if self.active_finger_count != PINKY_SPECIAL_COMMAND:
            self._reset_grasp_type7_rotation_state()
            return False

        qdot_max = self._active_finger_qdot_max(qdot)
        self.grasp_type7_last_qdot_max = qdot_max

        if self.grasp_type7_phase == "idle":
            self._start_grasp_type7_rotation_state()

        if self.grasp_type7_phase == "grasp_stabilizing":
            if qdot_max <= float(self.cfg.grasp_type7_start_qdot_threshold):
                if self.grasp_type7_stable_since is None:
                    self.grasp_type7_stable_since = now
                elif now - self.grasp_type7_stable_since >= float(self.cfg.grasp_type7_start_hold_sec):
                    self.grasp_type7_phase = "rotating"
                    self.grasp_type7_rotation_started_at = now
                    self.grasp_type7_done_since = None
                    self.grasp_type7_rotation_cg_ref = None
                    self._log(
                        "[GRASP_TYPE7] grasp stabilized -> rotation_start "
                        f"qdot_max={qdot_max:.4f}, "
                        f"hold={now - self.grasp_type7_stable_since:.2f}s"
                    )
            else:
                self.grasp_type7_stable_since = None
            return False

        if self.grasp_type7_phase == "rotating":
            rotation_elapsed = (
                0.0
                if self.grasp_type7_rotation_started_at is None
                else now - self.grasp_type7_rotation_started_at
            )
            if rotation_elapsed < float(self.cfg.grasp_type7_min_rotation_sec):
                self.grasp_type7_done_since = None
                return True

            if qdot_max <= float(self.cfg.grasp_type7_done_qdot_threshold):
                if self.grasp_type7_done_since is None:
                    self.grasp_type7_done_since = now
                elif now - self.grasp_type7_done_since >= float(self.cfg.grasp_type7_done_hold_sec):
                    self.grasp_type7_rotation_done = True
                    self._log(
                        "[GRASP_TYPE7] rotation_done detected "
                        f"qdot_max={qdot_max:.4f}, "
                        f"hold={now - self.grasp_type7_done_since:.2f}s, "
                        f"rotation_elapsed={rotation_elapsed:.2f}s"
                    )
                    if bool(self.cfg.grasp_type7_index_transition_enable):
                        self._start_grasp_type7_index_detach_pregrasp()
                        return False
                    self.grasp_type7_phase = "rotation_done"
            else:
                self.grasp_type7_done_since = None
            return True

        if self._is_grasp_type7_attach_phase():
            return self._update_grasp_type7_attach_state(qdot, now)

        if self.grasp_type7_phase == "rotation_done":
            return not bool(self.cfg.grasp_type7_stop_rotation_when_done)

        if self._is_grasp_type7_transition_phase() or self.grasp_type7_phase == "transition_done":
            return False

        return False

    def _reset_pinky_special_grasp(self):
        self.pinky_special_hold_pose = None
        self._reset_grasp_type7_rotation_state()

    def _start_pinky_special_grasp(self):
        pinky_idxs = np.asarray(FINGER_JOINT_INDEX[PINKY_FINGER_ID], dtype=int)
        target = self.hand_q[pinky_idxs].copy()
        target[:2] = PINKY_SPECIAL_FIXED_LOCAL_TARGETS

        self._clear_transition_state()
        self._reset_envelop_grasp()
        self.pinky_special_hold_pose = target
        self.use_fingers = selected_fingers(PINKY_SPECIAL_GRASP_COUNT)
        self.policy = GraspPolicy(self.use_fingers, self.cfg)
        self.active_finger_count = PINKY_SPECIAL_COMMAND
        self._start_grasp_type7_rotation_state()

        return target.copy()

    def _start_envelop_grasp(self, now):
        self._clear_transition_state()
        self._reset_pinky_special_grasp()
        self.envelop_hold_pose = self.hand_q.copy()
        self.envelop_started_at = now
        self.envelop_thumb_enabled = False
        self.envelop_thumb_start_at = None
        self.envelop_last_joint_stall_since = None
        self.envelop_last_info = {
            "active_non_thumb": [],
            "active_thumb": [],
            "thumb_enabled": False,
            "tau_level": float(self.cfg.alpha1 * self.cfg.envelop_tau_scale),
        }

    def _envelop_non_thumb_joint_start_time(self, finger, local_joint):
        # Joint-stage sequencing: all non-thumb 2nd joints start together,
        # then all 3rd joints, then all 4th joints. The `finger` argument is
        # intentionally unused except for API symmetry/readability.
        _ = finger
        joint_order_idx = ENVELOP_FINGER_TORQUE_LOCAL_JOINTS.index(local_joint)
        return self.envelop_started_at + joint_order_idx * self.cfg.envelop_joint_delay

    def _calc_envelop_grasp(self, qdot, now):
        if self.envelop_hold_pose is None or self.envelop_started_at is None:
            self._start_envelop_grasp(now)

        _ = qdot  # The envelop timing is intentionally time-based, not velocity/stall-based.

        tau = np.zeros(JOINT_COUNT, dtype=np.float64)
        pd = np.zeros(JOINT_COUNT, dtype=np.float64)
        hold_mask = np.ones(JOINT_COUNT, dtype=bool)
        active_non_thumb = []
        active_thumb = []

        # Constant torque level for each active envelop-grasp joint.
        # Default is deliberately lower than alpha1/4 because the previous value was too strong on hardware.
        tau_level = float(self.cfg.alpha1 * self.cfg.envelop_tau_scale)
        tau_level = min(tau_level, float(self.cfg.groped_tau_limit))

        # Time-only joint-stage sequencing:
        #   t0 + 0*delay: non-thumb 2nd joints
        #   t0 + 1*delay: non-thumb 3rd joints
        #   t0 + 2*delay: non-thumb 4th joints + thumb 3rd joint
        #   t0 + 3*delay: thumb 4th joint
        # No qdot/stall condition is used for thumb triggering.
        for finger in ENVELOP_FINGER_ORDER:
            idxs = FINGER_JOINT_INDEX[finger]
            for local_joint in ENVELOP_FINGER_TORQUE_LOCAL_JOINTS:
                joint_idx = idxs[local_joint]
                start_t = self._envelop_non_thumb_joint_start_time(finger, local_joint)
                if now >= start_t:
                    hold_mask[joint_idx] = False
                    tau[joint_idx] = float(self.cfg.envelop_non_thumb_tau_sign) * tau_level
                    active_non_thumb.append(joint_idx)

        thumb_idxs = FINGER_JOINT_INDEX[1]
        for order_idx, local_joint in enumerate(ENVELOP_THUMB_TORQUE_LOCAL_JOINTS):
            # Thumb starts one stage later than before:
            # thumb joint 3 starts with the four non-thumb 4th joints,
            # thumb joint 4 starts one joint-delay after that.
            stage_idx = order_idx + 2
            joint_idx = thumb_idxs[local_joint]
            start_t = self.envelop_started_at + stage_idx * self.cfg.envelop_joint_delay
            if now >= start_t:
                hold_mask[joint_idx] = False
                tau[joint_idx] = float(self.cfg.envelop_thumb_tau_sign) * tau_level
                active_thumb.append(joint_idx)

        self.envelop_thumb_enabled = len(active_thumb) > 0

        hold_idxs = np.flatnonzero(hold_mask)
        pd[hold_idxs], err = pose_pd(
            self.envelop_hold_pose[hold_idxs],
            self.hand_q[hold_idxs],
            qdot[hold_idxs],
            kp=self.cfg.pose_kp,
            kd=self.cfg.pose_kd,
            limit=self.cfg.pose_pd_limit,
        )

        self.envelop_last_info = {
            "active_non_thumb": active_non_thumb,
            "active_thumb": active_thumb,
            "thumb_enabled": self.envelop_thumb_enabled,
            "tau_level": tau_level,
        }
        return tau, pd, err

    def apply_pose_type(self, pose_type: int, now: float) -> None:
        if pose_type not in POSE_TYPE_TARGETS:
            raise ValueError("pose_type must be 1, 2, or 3")
        state, state_start, _ = self._apply_pose_type_command(pose_type, now)
        self.state = state
        self.state_start = state_start

    def _cancel_deferred_for_external_command(self) -> None:
        if self.deferred_finger_count is not None and not self.skip_add_pre_grasp_once:
            self.deferred_finger_count = None
            self.deferred_finger_count_at = None

    def apply_grasp_type(self, requested_count: int, now: float, *, internal: bool = False) -> None:
        if requested_count < -1 or requested_count > 7:
            raise ValueError("grasp_type must be one of -1, 0, 1, 2, 3, 4, 5, 6, 7")

        if not internal:
            self._cancel_deferred_for_external_command()

        command_count = requested_count
        if not self.skip_add_pre_grasp_once:
            self.blending_fingers = []
            self.blend_started_at = None
            self.blend_until = None

        if requested_count == -1:
            self.active_finger_count = 0
            self._clear_transition_state()
            self._reset_envelop_grasp()
            self._reset_pinky_special_grasp()
            self.state = "NORMAL_POSE"
            self.state_start = now
            self._log("[COMMAND] grasp_type=-1 -> NORMAL_POSE")
            return

        if requested_count == 0:
            self.active_finger_count = 0
            self._clear_transition_state()
            self._reset_envelop_grasp()
            self._reset_pinky_special_grasp()
            self.state = "PRE_GRASP_POSE"
            self.state_start = now
            self._log("[COMMAND] grasp_type=0 -> PRE_GRASP_POSE")
            return

        if requested_count == 6:
            self.active_finger_count = 6
            self._clear_transition_state()
            self._reset_pinky_special_grasp()
            self._start_envelop_grasp(now)
            self.state = "ENVELOP_GRASP"
            self.state_start = now
            self._log(
                "[COMMAND] grasp_type=6 -> ENVELOP_GRASP "
                f"tau_per_joint={self.cfg.alpha1 * self.cfg.envelop_tau_scale:.4f}, "
                "joint_stage_order=2 -> 3 -> (4+thumb3) -> thumb4"
            )
            return

        if requested_count == PINKY_SPECIAL_COMMAND:
            pinky_target = self._start_pinky_special_grasp()
            self.state = "GROPED_GRASP"
            self.state_start = now
            self._log(
                "[COMMAND] grasp_type=7 -> GROPED_GRASP as grasp_type=4, "
                f"use_fingers={self.use_fingers}, "
                f"pinky_target_rad={np.round(pinky_target, 4).tolist()}"
            )
            return

        self._reset_envelop_grasp()
        self._reset_pinky_special_grasp()

        if (
            self.active_finger_count in (1, 2)
            and requested_count in (1, 2)
            and requested_count != self.active_finger_count
        ):
            command_count = 3
            self.deferred_finger_count = requested_count
            self.deferred_finger_count_at = None
            self._log(f"[COMMAND] grasp_type={requested_count} requested, switch via 3")

        prev_fingers = (
            set(self.use_fingers)
            if 0 < self.active_finger_count <= 5
            else set()
        )
        target_fingers = selected_fingers(command_count)
        new_fingers = set(target_fingers)
        added = sorted(new_fingers - prev_fingers)

        if (
            0 < self.active_finger_count <= 5
            and added
            and not self.skip_add_pre_grasp_once
        ):
            self.adding_finger_target_count = command_count
            self.adding_pre_grasp_fingers = added
            self.adding_ready_at = now + ADD_FINGER_PRE_GRASP_DELAY
            self.state = "GROPED_GRASP"
            self.state_start = now
            self._log(
                f"[COMMAND] grasp_type={command_count} prepare added fingers "
                f"{added} at PRE_GRASP for {ADD_FINGER_PRE_GRASP_DELAY:.2f}s"
            )
            return

        self.skip_add_pre_grasp_once = False
        self.use_fingers = target_fingers
        self.policy = GraspPolicy(self.use_fingers, self.cfg)
        self.active_finger_count = command_count
        self.adding_finger_target_count = None
        self.adding_pre_grasp_fingers = []
        self.adding_ready_at = None
        self.state = "GROPED_GRASP"
        self.state_start = now

        removed = sorted(prev_fingers - new_fingers)
        if (
            self.deferred_finger_count is not None
            and self.deferred_finger_count_at is None
            and command_count == 3
        ):
            self.deferred_finger_count_at = now + FINGER_SWITCH_VIA_THREE_DELAY
            self._log(
                f"[COMMAND] hold grasp_type=3 for {FINGER_SWITCH_VIA_THREE_DELAY:.2f}s "
                f"before grasp_type={self.deferred_finger_count}"
            )

        self._log(
            f"[COMMAND] grasp_type={command_count} -> GROPED_GRASP, "
            f"use_fingers={self.use_fingers}, added={added}, removed={removed}"
        )

    def _process_timers(self, now: float) -> None:
        if (
            self.adding_finger_target_count is not None
            and self.adding_ready_at is not None
            and now >= self.adding_ready_at
        ):
            command = self.adding_finger_target_count
            self.blending_fingers = []
            self.blend_started_at = None
            self.blend_until = None
            self.adding_finger_target_count = None
            self.adding_pre_grasp_fingers = []
            self.adding_ready_at = None
            self.skip_add_pre_grasp_once = True
            self.apply_grasp_type(command, now, internal=True)
            return

        if (
            self.deferred_finger_count is not None
            and self.deferred_finger_count_at is not None
            and self.adding_finger_target_count is None
            and now >= self.deferred_finger_count_at
        ):
            command = self.deferred_finger_count
            self.deferred_finger_count = None
            self.deferred_finger_count_at = None
            self.apply_grasp_type(command, now, internal=True)

    def step(self, q: np.ndarray, qdot: np.ndarray, now: float) -> ControlOutput:
        q = np.asarray(q, dtype=np.float64)
        qdot = np.asarray(qdot, dtype=np.float64)
        if q.shape != (JOINT_COUNT,) or qdot.shape != (JOINT_COUNT,):
            raise ValueError(f"q and qdot must have shape ({JOINT_COUNT},)")

        self.sync_joint_state(q)
        if self.state_start == 0.0:
            self.state_start = now
        self._process_timers(now)

        err = np.zeros(JOINT_COUNT, dtype=np.float64)
        grasp_tau = np.zeros(JOINT_COUNT, dtype=np.float64)
        inactive_pd = np.zeros(JOINT_COUNT, dtype=np.float64)
        alpha = {}
        cg = np.zeros(3, dtype=np.float64)
        cv = np.zeros(3, dtype=np.float64)
        rotation_enabled = False
        self.inactive_pd_target[:] = np.nan

        if self.state == "NORMAL_POSE":
            tau, err = pose_pd(
                HAND_NORMAL_POSE,
                self.hand_q,
                qdot,
                kp=self.cfg.pose_kp,
                kd=self.cfg.pose_kd,
                limit=self.cfg.pose_pd_limit,
            )

        elif self.state == "PRE_GRASP_POSE":
            tau, err = pose_pd(
                self._current_pre_grasp_pose(),
                self.hand_q,
                qdot,
                kp=self.cfg.pose_kp,
                kd=self.cfg.pose_kd,
                limit=self.cfg.pose_pd_limit,
            )

        elif self.state == "GROPED_GRASP":
            if self.active_finger_count == PINKY_SPECIAL_COMMAND:
                rotation_enabled = (
                    self.cfg.rotation_enable_for_grasp_type7
                    and self._update_grasp_type7_rotation_state(qdot, now)
                )

            rotation_center_ref = None
            center_hold_enabled = False
            if self.active_finger_count == PINKY_SPECIAL_COMMAND and rotation_enabled:
                if self.grasp_type7_rotation_cg_ref is None:
                    _, _, cg_ref, _, _ = self.policy.calc_alpha_and_forces(self.hand_q)
                    self.grasp_type7_rotation_cg_ref = cg_ref.copy()
                    self._log(
                        "[GRASP_TYPE7] capture rotation geometric centroid ref "
                        f"cg_ref={np.round(self.grasp_type7_rotation_cg_ref, 4).tolist()}"
                    )
                rotation_center_ref = self.grasp_type7_rotation_cg_ref
                center_hold_enabled = bool(self.cfg.grasp_type7_center_hold_enable)

            grasp_tau, alpha, cg, cv, _ = self.policy.calc_grasp_tau(
                self.hand_q,
                rotation_enabled=rotation_enabled,
                rotation_center=rotation_center_ref,
                center_hold_target=rotation_center_ref,
                center_hold_enabled=center_hold_enabled,
            )

            blend_pd = np.zeros(JOINT_COUNT, dtype=np.float64)
            if (
                self.blending_fingers
                and self.blend_started_at is not None
                and self.blend_until is not None
            ):
                duration = max(self.blend_until - self.blend_started_at, 1e-6)
                blend_raw = np.clip((now - self.blend_started_at) / duration, 0.0, 1.0)
                blend = blend_raw * blend_raw * (3.0 - 2.0 * blend_raw)
                for finger in list(self.blending_fingers):
                    idxs = np.asarray(FINGER_JOINT_INDEX[finger], dtype=int)
                    pd, _ = pose_pd(
                        self._current_pre_grasp_pose()[idxs],
                        self.hand_q[idxs],
                        qdot[idxs],
                        kp=self.cfg.pose_kp,
                        kd=self.cfg.pose_kd,
                        limit=self.cfg.pose_pd_limit,
                    )
                    grasp_tau[idxs] *= blend
                    blend_pd[idxs] = (1.0 - blend) * pd
                if blend_raw >= 1.0:
                    self.blending_fingers = []
                    self.blend_started_at = None
                    self.blend_until = None

            g7_transition_pd = np.zeros(JOINT_COUNT, dtype=np.float64)
            g7_transition_attach_tau = np.zeros(JOINT_COUNT, dtype=np.float64)
            inactive_control_fingers = list(self.use_fingers)
            if (
                self.active_finger_count == PINKY_SPECIAL_COMMAND
                and self._is_grasp_type7_transition_phase()
            ):
                if self.grasp_type7_transition_finger_id is not None:
                    inactive_control_fingers.append(self.grasp_type7_transition_finger_id)
                g7_transition_pd = self._calc_grasp_type7_transition_pd(qdot)
                g7_transition_attach_tau = self._calc_grasp_type7_transition_attach_tau(cv)

            inactive_pd = self._inactive_pre_grasp_pd(
                inactive_control_fingers,
                qdot,
                self.adding_pre_grasp_fingers,
            )
            tau = (
                grasp_tau
                + inactive_pd
                + blend_pd
                + g7_transition_pd
                + g7_transition_attach_tau
            )

        elif self.state == "ENVELOP_GRASP":
            grasp_tau, envelop_pd, err = self._calc_envelop_grasp(qdot, now)
            tau = grasp_tau + envelop_pd

        else:
            tau = np.zeros(JOINT_COUNT, dtype=np.float64)

        return ControlOutput(
            tau=np.asarray(tau, dtype=np.float64),
            state=self.state,
            state_elapsed=max(0.0, now - self.state_start),
            err=err.copy(),
            grasp_tau=grasp_tau.copy(),
            inactive_pd=inactive_pd.copy(),
            alpha={int(k): float(v) for k, v in alpha.items()},
            cg=cg.copy(),
            cv=cv.copy(),
            rotation_enabled=bool(rotation_enabled),
            use_fingers=list(self.use_fingers),
            active_finger_count=int(self.active_finger_count),
            inactive_pd_target=self.inactive_pd_target.copy(),
            envelop_info=dict(self.envelop_last_info),
            g7_phase=str(self.grasp_type7_phase),
        )
