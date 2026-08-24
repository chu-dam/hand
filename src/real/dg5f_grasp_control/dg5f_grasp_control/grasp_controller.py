from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional

import numpy as np

from dg5f_grasp_control.config import RuntimeConfig
from dg5f_grasp_control.control_utils import pose_pd
from dg5f_grasp_control.grasp_policy import (
    ALPHA_DISTRIBUTION_THUMB_DISTANCE_PROPORTIONAL,
    BASE_JOINT_TAU_LIMIT,
    GraspPolicy,
    GraspPolicyResult,
)
from dg5f_grasp_control.hand_model import (
    FINGER_AVOIDANCE_JOINT_LOCAL_INDEX,
    FINGER_JOINT_INDEX,
    GRASP_TAU_SIGN,
    JOINT_COUNT,
    get_finger_avoidance_joint_limits,
    selected_fingers,
)
from dg5f_grasp_control.kinematics import (
    tip_jacobian,
    tip_position,
)
from dg5f_grasp_control.poses import (
    RIGHT_HAND_CONTINUOUS_ROTATION_POSE,
    get_pose_type_targets,
)

FINGER_SWITCH_VIA_THREE_DELAY = 0.5

ENVELOP_FINGER_ORDER = [2, 3, 4, 5]
ENVELOP_FINGER_TORQUE_LOCAL_JOINTS = [1, 2, 3]
ENVELOP_PINKY_TORQUE_LOCAL_JOINTS = [2, 3]
ENVELOP_THUMB_TORQUE_LOCAL_JOINTS = [2, 3]

PINKY_FINGER_ID = 5
INACTIVE_COLLISION_CHAIN = (2, 3, 4, 5)
CARD_THUMB_ID = 1
CARD_INDEX_ID = 2
CONTINUOUS_ROTATION_GROUPS = ((3,), (4, 2), (1,), (5,))
CONTINUOUS_ROTATION_GROUP_NAMES = ("middle", "ring_index", "thumb", "pinky")
CONTINUOUS_ROTATION_RELEASE_JOINTS = {
    1: ((1, -1.0), (2, -1.0)),
    2: ((1, -1.0), (2, 1.0)),
    3: ((1, -1.0), (2, 1.0)),
    4: ((1, -1.0), (2, 1.0)),
    5: ((0, -1.0),),
}


def _copy_finger_vectors(values: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    return {
        int(finger): np.asarray(vector, dtype=np.float64).copy()
        for finger, vector in values.items()
    }


@dataclass
class ControlOutput:
    tau: np.ndarray
    state: str
    state_elapsed: float
    err: np.ndarray = field(default_factory=lambda: np.zeros(JOINT_COUNT, dtype=np.float64))
    grasp_tau: np.ndarray = field(default_factory=lambda: np.zeros(JOINT_COUNT, dtype=np.float64))
    translation_torques: np.ndarray = field(
        default_factory=lambda: np.zeros(JOINT_COUNT, dtype=np.float64)
    )
    inactive_pd: np.ndarray = field(default_factory=lambda: np.zeros(JOINT_COUNT, dtype=np.float64))
    alpha: Dict[int, float] = field(default_factory=dict)
    cg: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    cv: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    fingertip_positions: Dict[int, np.ndarray] = field(default_factory=dict)
    grasp_forces: Dict[int, np.ndarray] = field(default_factory=dict)
    translation_forces: Dict[int, np.ndarray] = field(default_factory=dict)
    rotation_forces: Dict[int, np.ndarray] = field(default_factory=dict)
    center_hold_forces: Dict[int, np.ndarray] = field(default_factory=dict)
    collision_forces: Dict[int, np.ndarray] = field(default_factory=dict)
    total_forces: Dict[int, np.ndarray] = field(default_factory=dict)
    use_fingers: List[int] = field(default_factory=list)
    active_finger_count: int = 0
    inactive_pd_target: np.ndarray = field(
        default_factory=lambda: np.full(JOINT_COUNT, np.nan, dtype=np.float64)
    )
    inactive_collision_min_clearance_m: float = -1.0
    inactive_collision_avoidance_offsets_rad: np.ndarray = field(
        default_factory=lambda: np.zeros(5, dtype=np.float64)
    )
    inactive_collision_avoidance_active: List[bool] = field(
        default_factory=lambda: [False] * 5
    )
    envelop_info: Dict[str, object] = field(default_factory=dict)
    relative_rotation_phase: str = "idle"
    relative_rotation_target_rad: float = 0.0
    relative_rotation_current_rad: float = 0.0
    relative_rotation_error_rad: float = 0.0
    relative_rotation_angular_velocity: float = 0.0
    relative_rotation_command_moment: float = 0.0
    relative_rotation_axis: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    relative_rotation_pivot: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    relative_rotation_control_mode: str = "idle"
    relative_translation_phase: str = "idle"
    relative_translation_start_centroid: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    relative_translation_target_centroid: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    relative_translation_delta: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    relative_translation_error: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    relative_translation_centroid_velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    relative_translation_command_force: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    relative_translation_torque_target: float = 0.0
    relative_translation_force_scale: float = 0.0
    relative_translation_control_mode: str = "idle"
    relative_translation_dls_sigma_min: float = 0.0
    relative_translation_dls_condition: float = 0.0
    relative_translation_joint_error: np.ndarray = field(
        default_factory=lambda: np.zeros(JOINT_COUNT, dtype=np.float64)
    )
    relative_translation_position_torques: np.ndarray = field(
        default_factory=lambda: np.zeros(JOINT_COUNT, dtype=np.float64)
    )
    relative_translation_nullspace_grasp_torques: np.ndarray = field(
        default_factory=lambda: np.zeros(JOINT_COUNT, dtype=np.float64)
    )
    effective_thumb_centroid_bias: float = 0.0
    relative_rotation_force_balance_blend: float = 0.0


class GraspController:
    """Hardware-independent DG5F grasp state machine and torque controller.

    This class owns the grasp/pose state, finger switching, groped-grasp
    policy, inactive-finger PD, envelop grasp, and Cartesian manipulation.
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
        self.pose_type_targets = get_pose_type_targets(cfg.hand_side)
        self.finger_avoidance_joint_limits = (
            get_finger_avoidance_joint_limits(cfg.hand_side)
        )

        self.pose_type = 1
        self.pre_grasp_pose_type = 2
        self.state = "NORMAL_POSE"
        self.state_start = 0.0
        self.active_finger_count = 0

        self.deferred_finger_count = None
        self.deferred_finger_count_at = None

        self.inactive_pd_target = np.full(JOINT_COUNT, np.nan, dtype=np.float64)
        self.inactive_collision_avoidance_offsets_rad = np.zeros(
            5,
            dtype=np.float64,
        )
        self.inactive_collision_avoidance_active = [False] * 5
        self.inactive_collision_follow_source = np.zeros(
            5,
            dtype=np.int64,
        )
        self.inactive_collision_follow_offset_rad = np.zeros(
            5,
            dtype=np.float64,
        )
        self.inactive_collision_approach_direction = np.zeros(
            5,
            dtype=np.float64,
        )
        self.inactive_collision_previous_joint_positions = None
        self.inactive_collision_min_clearance_m = -1.0

        self.envelop_hold_pose = None
        self.envelop_started_at = None
        self.envelop_thumb_enabled = False
        self.envelop_thumb_start_at = None
        self.envelop_last_joint_stall_since = None
        self.envelop_last_info = {}

        # Generic grasp types (1..5) already use Cv == Cg and balanced grasp
        # forces. Relative rotation stores the command-time contact geometry
        # and drives non-thumb fingertips around the current thumb pivot.
        self.relative_rotation_phase = "idle"
        self.relative_rotation_target_rad = 0.0
        self.relative_rotation_started_at = None
        self.relative_rotation_pivot = np.zeros(3, dtype=np.float64)
        self.relative_rotation_start_fingertips = {}
        self.relative_rotation_current_rad = 0.0
        self.relative_rotation_error_rad = 0.0
        self.relative_rotation_angular_velocity = 0.0
        self.relative_rotation_command_moment = 0.0
        self.relative_rotation_axis = np.zeros(3, dtype=np.float64)
        self.relative_rotation_last_wrapped_angle = None
        self.relative_rotation_reference_progress = 0.0
        self.continuous_rotation_active = False
        self.continuous_rotation_phase = "idle"
        self.continuous_rotation_phase_started_at = None
        self.continuous_rotation_group_index = 0
        self.continuous_rotation_pose_target = None
        # Relative translation stores each fingertip position at command time
        # and tracks the translated targets with Cartesian PD forces.
        self.relative_translation_phase = "idle"
        self.relative_translation_start_centroid = np.zeros(3, dtype=np.float64)
        self.relative_translation_target_centroid = np.zeros(3, dtype=np.float64)
        self.relative_translation_delta = np.zeros(3, dtype=np.float64)
        self.relative_translation_error = np.zeros(3, dtype=np.float64)
        self.relative_translation_centroid_velocity = np.zeros(3, dtype=np.float64)
        self.relative_translation_command_force = np.zeros(3, dtype=np.float64)
        self.relative_translation_shape_forces = {}
        self.relative_translation_start_fingertips = {}
        self.relative_translation_target_fingertips = {}
        self.relative_translation_fingertip_velocities = {}
        self.relative_translation_contact_weights = {}
        self.relative_translation_max_axis_fingertip_error = 0.0
        self.relative_translation_reference_progress = 0.0
        self.relative_translation_control_axis_error = 0.0
        self.relative_translation_control_axis_drive_force = 0.0
        self.relative_translation_started_at = None
        self.relative_translation_reached_since = None
        self.regular_force_balance_error_started_at = None
        self.last_regular_policy_result = None
        self.last_regular_policy_fingers = ()

        self.rotation_hand_to_world = np.eye(3, dtype=np.float64)
        self.card_phase = "idle"
        self.card_phase_started_at = None
        self.card_stall_reference_positions = {}
        self.card_stall_since = None
        self.card_stall_max_motion_m = 0.0
        self.card_stable_elapsed_sec = 0.0
        self.card_floor_stall_since = {}
        self.card_floor_stable_elapsed_sec = {}
        self.card_floor_motion_m = {}
        self.card_floor_contact_detected = {}
        self.card_floor_contact_positions = {}
        self.card_thumb_j1_hold_target = None
        self.card_thumb_j1_hold_error_rad = 0.0
        self.card_thumb_j1_hold_tau = 0.0
        self.card_index_flex_hold_target = None
        self.card_index_tip_reached_since = None

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

    def cancel_relative_rotation(self) -> None:
        """Cancel the stored generic relative-rotation request."""

        self.relative_rotation_phase = "idle"
        self.relative_rotation_target_rad = 0.0
        self.relative_rotation_started_at = None
        self.relative_rotation_pivot[:] = 0.0
        self.relative_rotation_start_fingertips = {}
        self.relative_rotation_current_rad = 0.0
        self.relative_rotation_error_rad = 0.0
        self.relative_rotation_angular_velocity = 0.0
        self.relative_rotation_command_moment = 0.0
        self.relative_rotation_axis[:] = 0.0
        self.relative_rotation_last_wrapped_angle = None
        self.relative_rotation_reference_progress = 0.0

    def cancel_continuous_rotation(self) -> None:
        self.continuous_rotation_active = False
        self.continuous_rotation_phase = "idle"
        self.continuous_rotation_phase_started_at = None
        self.continuous_rotation_group_index = 0
        self.continuous_rotation_pose_target = None

    def _set_continuous_rotation_phase(self, phase: str, now: float) -> None:
        self.continuous_rotation_phase = str(phase)
        self.continuous_rotation_phase_started_at = float(now)
        self._log(f"[CONTINUOUS_ROTATION] phase={phase}")

    def _start_continuous_release(self, now: float) -> None:
        group = CONTINUOUS_ROTATION_GROUPS[
            self.continuous_rotation_group_index
        ]
        target = self.continuous_rotation_pose_target.copy()
        if group == (5,):
            for finger in range(1, 5):
                indices = np.asarray(FINGER_JOINT_INDEX[finger], dtype=int)
                target[indices] = self.pose_type_targets[5][indices]
        elif group == (3,):
            pinky_j1 = int(FINGER_JOINT_INDEX[5][0])
            target[pinky_j1] = self.pose_type_targets[5][pinky_j1]
        for finger in group:
            if finger in (2, 3, 4):
                joint_1 = int(FINGER_JOINT_INDEX[finger][0])
                target[joint_1] = RIGHT_HAND_CONTINUOUS_ROTATION_POSE[joint_1]
            release_deg = (
                self.cfg.continuous_rotation_index_ring_release_deg
                if finger in (2, 4)
                else self.cfg.continuous_rotation_release_deg
            )
            for local_joint, direction in CONTINUOUS_ROTATION_RELEASE_JOINTS[finger]:
                joint_release_deg = (
                    self.cfg.continuous_rotation_ring_j2_release_deg
                    if finger == 4 and local_joint == 1
                    else self.cfg.continuous_rotation_thumb_j2_release_deg
                    if finger == 1 and local_joint == 1
                    else release_deg
                )
                release_angle = np.deg2rad(float(joint_release_deg))
                joint = int(FINGER_JOINT_INDEX[finger][local_joint])
                target[joint] += direction * release_angle
        self.continuous_rotation_pose_target = target
        name = CONTINUOUS_ROTATION_GROUP_NAMES[
            self.continuous_rotation_group_index
        ]
        self._set_continuous_rotation_phase(f"continuous_release_{name}", now)

    def _start_continuous_move(self, now: float) -> None:
        group = CONTINUOUS_ROTATION_GROUPS[
            self.continuous_rotation_group_index
        ]
        for finger in group:
            indices = np.asarray(FINGER_JOINT_INDEX[finger], dtype=int)
            self.continuous_rotation_pose_target[indices] = (
                RIGHT_HAND_CONTINUOUS_ROTATION_POSE[indices]
            )
        name = CONTINUOUS_ROTATION_GROUP_NAMES[
            self.continuous_rotation_group_index
        ]
        self._set_continuous_rotation_phase(f"continuous_move_{name}", now)

    def _process_continuous_rotation(self, now: float) -> None:
        if not self.continuous_rotation_active:
            return
        elapsed = float(now) - float(self.continuous_rotation_phase_started_at)
        if self.continuous_rotation_phase.startswith("continuous_release_"):
            if elapsed >= float(self.cfg.continuous_rotation_release_sec):
                if self.continuous_rotation_group_index == 3:
                    self.continuous_rotation_group_index = 0
                    self._start_continuous_release(now)
                else:
                    self._start_continuous_move(now)
            return
        if self.continuous_rotation_phase.startswith("continuous_move_"):
            if elapsed < float(self.cfg.continuous_rotation_move_sec):
                return
            self.continuous_rotation_group_index += 1
            self._start_continuous_release(now)
            return

    def start_continuous_rotation(self, now: float) -> bool:
        if (
            self.cfg.hand_side != "right"
            or self.state != "PRE_GRASP_POSE"
            or self.pose_type != 5
        ):
            self._log(
                "[CONTINUOUS_ROTATION] ignored: requires right-hand "
                "PRE_GRASP_POSE with pose_type=5"
            )
            return False
        self.continuous_rotation_active = True
        self.continuous_rotation_group_index = 0
        self.continuous_rotation_pose_target = self.pose_type_targets[5].copy()
        self._start_continuous_release(now)
        return True

    def stop_continuous_rotation(self, now: float) -> None:
        self.cancel_continuous_rotation()
        self._log("[CONTINUOUS_ROTATION] stopped")

    def cancel_relative_translation(self) -> None:
        """Cancel the stored Cartesian fingertip translation target."""

        self.relative_translation_phase = "idle"
        self.relative_translation_start_centroid[:] = 0.0
        self.relative_translation_target_centroid[:] = 0.0
        self.relative_translation_delta[:] = 0.0
        self.relative_translation_error[:] = 0.0
        self.relative_translation_centroid_velocity[:] = 0.0
        self.relative_translation_command_force[:] = 0.0
        self.relative_translation_shape_forces = {}
        self.relative_translation_start_fingertips = {}
        self.relative_translation_target_fingertips = {}
        self.relative_translation_fingertip_velocities = {}
        self.relative_translation_contact_weights = {}
        self.relative_translation_max_axis_fingertip_error = 0.0
        self.relative_translation_reference_progress = 0.0
        self.relative_translation_control_axis_error = 0.0
        self.relative_translation_control_axis_drive_force = 0.0
        self.relative_translation_started_at = None
        self.relative_translation_reached_since = None

    def set_rotation_hand_to_world(self, rotation: np.ndarray) -> None:
        rotation = np.asarray(rotation, dtype=np.float64)
        if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
            raise ValueError("hand-to-world rotation must be a finite 3x3 matrix")
        self.rotation_hand_to_world = rotation.copy()

    def cancel_card_grasp(self) -> None:
        self.card_phase = "idle"
        self.card_phase_started_at = None
        self.card_stall_reference_positions = {}
        self.card_stall_since = None
        self.card_stall_max_motion_m = 0.0
        self.card_stable_elapsed_sec = 0.0
        self.card_floor_stall_since = {}
        self.card_floor_stable_elapsed_sec = {}
        self.card_floor_motion_m = {}
        self.card_floor_contact_detected = {}
        self.card_floor_contact_positions = {}
        self.card_thumb_j1_hold_target = None
        self.card_thumb_j1_hold_error_rad = 0.0
        self.card_thumb_j1_hold_tau = 0.0
        self.card_index_flex_hold_target = None
        self.card_index_tip_reached_since = None

    def _reset_regular_force_balance_state(self) -> None:
        self.regular_force_balance_error_started_at = None
        self.last_regular_policy_result = None
        self.last_regular_policy_fingers = ()

    @staticmethod
    def _scaled_policy_result(
        result: GraspPolicyResult,
        scale: float,
    ) -> GraspPolicyResult:
        scale = float(np.clip(scale, 0.0, 1.0))

        def scaled_vectors(values):
            return {
                int(finger): scale * np.asarray(vector, dtype=np.float64)
                for finger, vector in values.items()
            }

        return replace(
            result,
            tau=scale * np.asarray(result.tau, dtype=np.float64),
            alpha={
                int(finger): scale * float(value)
                for finger, value in result.alpha.items()
            },
            cg=np.asarray(result.cg, dtype=np.float64).copy(),
            cv=np.asarray(result.cg, dtype=np.float64).copy(),
            fingertip_positions=_copy_finger_vectors(
                result.fingertip_positions
            ),
            grasp_forces=scaled_vectors(result.grasp_forces),
            rotation_forces=scaled_vectors(result.rotation_forces),
            center_hold_forces=scaled_vectors(result.center_hold_forces),
            collision_forces=scaled_vectors(result.collision_forces),
            total_forces=scaled_vectors(result.total_forces),
        )

    def _regular_force_balance_fail_closed_result(
        self,
        now: float,
    ) -> GraspPolicyResult:
        start = self.regular_force_balance_error_started_at
        duration = float(self.cfg.force_balance_error_ramp_sec)
        if (
            start is None
            or not np.isfinite(start)
            or not np.isfinite(now)
            or not np.isfinite(duration)
            or duration <= 0.0
        ):
            scale = 0.0
        else:
            scale = float(np.clip(1.0 - (now - start) / duration, 0.0, 1.0))
            if scale <= 1e-12:
                scale = 0.0

        cache_is_usable = (
            self.last_regular_policy_result is not None
            and self.last_regular_policy_fingers == tuple(self.use_fingers)
        )
        if cache_is_usable and scale > 0.0:
            scaled = self._scaled_policy_result(
                self.last_regular_policy_result,
                scale,
            )
            current_geometry = self.policy.calc_zero_grasp_result(self.hand_q)
            return replace(
                scaled,
                tau=self.policy.calc_tau_from_total_forces(
                    self.hand_q,
                    scaled.total_forces,
                ),
                cg=current_geometry.cg.copy(),
                cv=current_geometry.cv.copy(),
                fingertip_positions=_copy_finger_vectors(
                    current_geometry.fingertip_positions
                ),
            )
        return self.policy.calc_zero_grasp_result(self.hand_q)

    def prepare_relative_rotation(
        self,
        angle_rad: float,
        now: float,
        *,
        internal: bool = False,
    ) -> bool:
        """Start closed-loop rotation relative to the contact configuration."""

        angle_rad = float(angle_rad)
        now = float(now)
        if not np.isfinite(angle_rad) or angle_rad == 0.0:
            raise ValueError("relative rotation angle must be finite and non-zero")
        maximum = float(np.deg2rad(self.cfg.relative_rotation_max_abs_deg))
        if not np.isfinite(maximum) or maximum <= 0.0:
            raise ValueError("relative rotation maximum must be finite and > 0")
        if abs(angle_rad) > maximum + 1e-12:
            raise ValueError(
                "relative rotation magnitude must be <= "
                f"{self.cfg.relative_rotation_max_abs_deg:.3f} deg"
            )
        if not np.isfinite(now):
            raise ValueError("now must be finite")
        if self.continuous_rotation_active and not internal:
            self._log(
                "[RELATIVE_ROTATION] ignored: continuous rotation is active"
            )
            return False
        if self.state != "GROPED_GRASP" or self.active_finger_count not in range(1, 6):
            self._log(
                "[RELATIVE_ROTATION] ignored: requires active grasp_type 1..5 "
                f"(state={self.state}, grasp_type={self.active_finger_count})"
            )
            return False
        if self.deferred_finger_count is not None:
            self._log(
                "[RELATIVE_ROTATION] ignored: finger composition transition "
                "is still active"
            )
            return False
        if (
            self.regular_force_balance_error_started_at is not None
            or self.relative_rotation_phase == "force_balance_error"
        ):
            self._log(
                "[RELATIVE_ROTATION] ignored: regular force balance is in "
                "fail-closed state; select the grasp again after checking "
                "the hand geometry"
            )
            return False
        if (
            self.last_regular_policy_result is None
            or self.last_regular_policy_fingers != tuple(self.use_fingers)
        ):
            self._log(
                "[RELATIVE_ROTATION] ignored: waiting for a successful "
                "regular force-balance control cycle"
            )
            return False

        start_fingertips = _copy_finger_vectors(
            self.last_regular_policy_result.fingertip_positions
        )
        if (
            set(start_fingertips) != set(self.use_fingers)
            or 1 not in start_fingertips
        ):
            self._log(
                "[RELATIVE_ROTATION] ignored: contact geometry is incomplete"
            )
            return False
        axis = np.array(
            [
                self.cfg.rotation_palm_normal_x,
                self.cfg.rotation_palm_normal_y,
                self.cfg.rotation_palm_normal_z,
            ],
            dtype=np.float64,
        )
        axis_norm = float(np.linalg.norm(axis))
        if not np.isfinite(axis_norm) or axis_norm <= 1e-12:
            raise ValueError("rotation palm-normal axis must be finite and non-zero")
        axis /= axis_norm

        self.cancel_relative_translation()
        self.cancel_relative_rotation()
        self.relative_rotation_target_rad = angle_rad
        self.relative_rotation_pivot = start_fingertips[1].copy()
        self.relative_rotation_start_fingertips = start_fingertips
        self.relative_rotation_current_rad = 0.0
        self.relative_rotation_error_rad = angle_rad
        self.relative_rotation_angular_velocity = 0.0
        self.relative_rotation_command_moment = 0.0
        self.relative_rotation_axis = axis
        self.relative_rotation_last_wrapped_angle = 0.0
        self.relative_rotation_reference_progress = 0.0
        self.relative_rotation_phase = "rotating"
        self.relative_rotation_started_at = now
        self._log(
            "[RELATIVE_ROTATION] closed-loop contact rotation started "
            f"target_delta_deg={np.degrees(angle_rad):.3f}, "
            f"axis={np.round(axis, 4).tolist()}"
        )
        return True

    def prepare_relative_translation(self, delta_hand: np.ndarray, now: float) -> bool:
        """Start a relative Cartesian target for every active fingertip."""

        delta_hand = np.asarray(delta_hand, dtype=np.float64)
        now = float(now)
        if delta_hand.shape != (3,) or not np.all(np.isfinite(delta_hand)):
            raise ValueError("relative translation must be a finite 3-vector")
        distance = float(np.linalg.norm(delta_hand))
        maximum = float(self.cfg.relative_translation_max_m)
        if not np.isfinite(maximum) or maximum <= 0.0:
            raise ValueError("relative translation maximum must be finite and > 0")
        if distance <= 0.0 or distance > maximum + 1e-12:
            raise ValueError(
                "relative translation norm must be within "
                f"(0, {maximum:.6f}] m"
            )
        if not np.isfinite(now):
            raise ValueError("now must be finite")
        if self.continuous_rotation_active:
            self._log(
                "[RELATIVE_TRANSLATION] ignored: continuous rotation is active"
            )
            return False
        if self.state != "GROPED_GRASP" or self.active_finger_count not in range(1, 6):
            self._log(
                "[RELATIVE_TRANSLATION] ignored: requires active grasp_type 1..5 "
                f"(state={self.state}, grasp_type={self.active_finger_count})"
            )
            return False
        if self.deferred_finger_count is not None:
            self._log(
                "[RELATIVE_TRANSLATION] ignored: finger composition transition "
                "is still active"
            )
            return False
        if self.regular_force_balance_error_started_at is not None:
            self._log(
                "[RELATIVE_TRANSLATION] ignored: regular force balance is in "
                "fail-closed state"
            )
            return False
        if (
            self.last_regular_policy_result is None
            or self.last_regular_policy_fingers != tuple(self.use_fingers)
        ):
            self._log(
                "[RELATIVE_TRANSLATION] ignored: waiting for a successful "
                "regular force-balance control cycle"
            )
            return False

        start = np.asarray(self.last_regular_policy_result.cg, dtype=np.float64)
        if start.shape != (3,) or not np.all(np.isfinite(start)):
            raise ValueError("current geometric centroid must be a finite 3-vector")

        start_fingertips = {
            int(finger): np.asarray(position, dtype=np.float64).copy()
            for finger, position in (
                self.last_regular_policy_result.fingertip_positions.items()
            )
        }
        if set(start_fingertips) != set(self.use_fingers) or any(
            position.shape != (3,) or not np.all(np.isfinite(position))
            for position in start_fingertips.values()
        ):
            self._log(
                "[RELATIVE_TRANSLATION] ignored: active fingertip geometry "
                "is incomplete or invalid"
            )
            return False
        contact_weights = {
            finger: 1.0 / len(start_fingertips)
            for finger in start_fingertips
        }

        self.cancel_relative_rotation()
        self.relative_translation_start_centroid = start.copy()
        self.relative_translation_delta = delta_hand.copy()
        self.relative_translation_target_centroid = start + delta_hand
        self.relative_translation_error = delta_hand.copy()
        self.relative_translation_centroid_velocity[:] = 0.0
        self.relative_translation_command_force[:] = 0.0
        self.relative_translation_shape_forces = {
            finger: np.zeros(3, dtype=np.float64)
            for finger in start_fingertips
        }
        self.relative_translation_start_fingertips = start_fingertips
        self.relative_translation_target_fingertips = {
            finger: position + delta_hand
            for finger, position in start_fingertips.items()
        }
        self.relative_translation_fingertip_velocities = {
            finger: np.zeros(3, dtype=np.float64)
            for finger in start_fingertips
        }
        self.relative_translation_contact_weights = contact_weights
        self.relative_translation_max_axis_fingertip_error = distance
        self.relative_translation_reference_progress = 0.0
        self.relative_translation_control_axis_error = 0.0
        self.relative_translation_control_axis_drive_force = 0.0
        self.relative_translation_started_at = now
        self.relative_translation_reached_since = None
        self.relative_translation_phase = "translating"
        self._log(
            "[RELATIVE_TRANSLATION] Cartesian fingertip target started "
            f"delta_link_base_mm={np.round(1000.0 * delta_hand, 3).tolist()}"
        )
        return True

    def _zero_translation_forces(self, tip_positions):
        return {
            int(finger): np.zeros(3, dtype=np.float64)
            for finger in tip_positions
        }

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return float(np.arctan2(np.sin(angle), np.cos(angle)))

    def _estimate_relative_rotation_angle(
        self,
        tip_positions: Dict[int, np.ndarray],
    ) -> float:
        """Estimate signed contact-constellation rotation about the saved axis."""

        axis = np.asarray(self.relative_rotation_axis, dtype=np.float64)
        weighted_sine = 0.0
        weighted_cosine = 0.0
        usable = 0
        radius_min = max(
            float(self.cfg.relative_rotation_radius_min),
            1e-6,
        )
        current_pivot = tip_positions.get(1)
        if current_pivot is None:
            raise ValueError("thumb pivot is unavailable")
        current_pivot = np.asarray(current_pivot, dtype=np.float64)
        for finger, start_position in self.relative_rotation_start_fingertips.items():
            if finger == 1:
                continue
            current_position = tip_positions.get(finger)
            if current_position is None:
                continue
            start_radius = (
                np.asarray(start_position, dtype=np.float64)
                - self.relative_rotation_pivot
            )
            # Remove any common translation of the thumb pivot when estimating
            # the object/contact-constellation angle.
            current_radius = (
                np.asarray(current_position, dtype=np.float64)
                - current_pivot
            )
            start_planar = start_radius - np.dot(start_radius, axis) * axis
            current_planar = current_radius - np.dot(current_radius, axis) * axis
            start_norm = float(np.linalg.norm(start_planar))
            current_norm = float(np.linalg.norm(current_planar))
            if start_norm < radius_min or current_norm < radius_min:
                continue
            sine = float(
                np.dot(
                    axis,
                    np.cross(start_planar, current_planar),
                )
                / (start_norm * current_norm)
            )
            cosine = float(
                np.dot(start_planar, current_planar)
                / (start_norm * current_norm)
            )
            weight = start_norm * current_norm
            weighted_sine += weight * sine
            weighted_cosine += weight * cosine
            usable += 1

        if usable == 0 or (
            abs(weighted_sine) <= 1e-15
            and abs(weighted_cosine) <= 1e-15
        ):
            raise ValueError("no usable contact radius for angle estimation")
        return float(np.arctan2(weighted_sine, weighted_cosine))

    def _calc_relative_rotation_forces(
        self,
        tip_positions: Dict[int, np.ndarray],
        now: float,
        qdot: np.ndarray | None = None,
    ) -> Dict[int, np.ndarray]:
        """Track non-thumb fingertip targets about the thumb pivot."""

        rotation_forces = self._zero_translation_forces(tip_positions)
        self.relative_rotation_command_moment = 0.0
        active_phases = {"rotating", "rotation_reached"}
        if self.relative_rotation_phase not in active_phases:
            return rotation_forces

        tip_positions = {
            int(finger): np.asarray(position, dtype=np.float64)
            for finger, position in tip_positions.items()
        }
        qdot = (
            np.zeros(JOINT_COUNT, dtype=np.float64)
            if qdot is None
            else np.asarray(qdot, dtype=np.float64)
        )
        now = float(now)
        expected_fingers = set(self.relative_rotation_start_fingertips)
        if (
            qdot.shape != (JOINT_COUNT,)
            or not np.all(np.isfinite(qdot))
            or not np.isfinite(now)
            or set(tip_positions) != expected_fingers
            or any(
                position.shape != (3,) or not np.all(np.isfinite(position))
                for position in tip_positions.values()
            )
        ):
            self.relative_rotation_phase = "rotation_error"
            self._log("[RELATIVE_ROTATION] stopped: invalid contact geometry/time")
            return rotation_forces

        try:
            wrapped_angle = self._estimate_relative_rotation_angle(tip_positions)
        except ValueError as exc:
            self.relative_rotation_phase = "rotation_error"
            self._log(f"[RELATIVE_ROTATION] stopped: {exc}")
            return rotation_forces

        if self.relative_rotation_last_wrapped_angle is not None:
            previous_wrapped = float(self.relative_rotation_last_wrapped_angle)
            angle_step = self._wrap_angle(wrapped_angle - previous_wrapped)
            self.relative_rotation_current_rad += angle_step
        else:
            self.relative_rotation_current_rad = wrapped_angle

        self.relative_rotation_last_wrapped_angle = wrapped_angle
        self.relative_rotation_error_rad = (
            self.relative_rotation_target_rad
            - self.relative_rotation_current_rad
        )

        elapsed = (
            0.0
            if self.relative_rotation_started_at is None
            else max(0.0, now - float(self.relative_rotation_started_at))
        )
        timeout = float(self.cfg.relative_rotation_timeout_sec)
        if (
            np.isfinite(timeout)
            and timeout > 0.0
            and elapsed >= timeout
        ):
            self.relative_rotation_phase = "rotation_timeout"
            self._log(
                "[RELATIVE_ROTATION] timeout; Cartesian rotation force removed "
                f"error_deg={np.degrees(self.relative_rotation_error_rad):.3f}"
            )
            return rotation_forces

        ramp_sec = max(
            0.0,
            float(self.cfg.relative_rotation_reference_ramp_sec),
        )
        if not np.isfinite(ramp_sec):
            self.relative_rotation_phase = "rotation_error"
            self._log("[RELATIVE_ROTATION] stopped: invalid reference ramp")
            return rotation_forces
        ramp_linear = 1.0 if ramp_sec <= 0.0 else np.clip(
            elapsed / ramp_sec,
            0.0,
            1.0,
        )
        reference_progress = float(
            ramp_linear * ramp_linear * (3.0 - 2.0 * ramp_linear)
        )
        self.relative_rotation_reference_progress = reference_progress
        reference_angle = reference_progress * self.relative_rotation_target_rad
        if ramp_sec > 0.0 and 0.0 < elapsed < ramp_sec:
            ramp_u = elapsed / ramp_sec
            reference_angular_velocity = (
                self.relative_rotation_target_rad
                * (6.0 * ramp_u * (1.0 - ramp_u) / ramp_sec)
            )
        else:
            reference_angular_velocity = 0.0

        alpha1 = float(self.cfg.alpha1)
        alpha1_reference = float(
            self.cfg.relative_rotation_alpha1_reference
        )
        alpha1_double_gain_scale = float(
            self.cfg.relative_rotation_alpha1_double_gain_scale
        )
        negative_direction_gain_scale = float(
            self.cfg.relative_rotation_negative_direction_gain_scale
        )
        positive_direction_damping_scale = float(
            self.cfg.relative_rotation_positive_direction_damping_scale
        )
        if (
            not np.isfinite(alpha1)
            or not np.isfinite(alpha1_reference)
            or not np.isfinite(alpha1_double_gain_scale)
            or not np.isfinite(negative_direction_gain_scale)
            or not np.isfinite(positive_direction_damping_scale)
            or alpha1 < 0.0
            or alpha1_reference <= 0.0
            or alpha1_double_gain_scale <= 1.0
            or negative_direction_gain_scale <= 0.0
            or positive_direction_damping_scale <= 0.0
        ):
            self.relative_rotation_phase = "rotation_error"
            self._log("[RELATIVE_ROTATION] stopped: invalid alpha1 gain scaling")
            return rotation_forces
        gain_exponent = float(np.log2(alpha1_double_gain_scale))
        gain_scale = float(
            np.power(alpha1 / alpha1_reference, gain_exponent)
        )
        if self.relative_rotation_target_rad < 0.0:
            gain_scale *= negative_direction_gain_scale
        position_kp = float(self.cfg.relative_rotation_position_kp) * gain_scale
        position_kd = float(self.cfg.relative_rotation_position_kd) * gain_scale
        if self.relative_rotation_target_rad > 0.0:
            position_kd *= positive_direction_damping_scale
        position_error_limit = float(
            self.cfg.relative_rotation_position_error_limit_m
        )
        position_tolerance = float(
            self.cfg.relative_rotation_position_tolerance_m
        )
        force_limit = float(self.cfg.relative_rotation_force_limit)
        radius_min = float(self.cfg.relative_rotation_radius_min)
        values = (
            position_kp,
            position_kd,
            position_error_limit,
            position_tolerance,
            force_limit,
            radius_min,
        )
        if (
            not all(np.isfinite(value) for value in values)
            or position_kp < 0.0
            or position_kd < 0.0
            or position_error_limit <= 0.0
            or position_tolerance < 0.0
            or force_limit < 0.0
            or radius_min <= 0.0
        ):
            self.relative_rotation_phase = "rotation_error"
            self._log("[RELATIVE_ROTATION] stopped: invalid rotation gains")
            return rotation_forces

        axis = np.asarray(self.relative_rotation_axis, dtype=np.float64)
        start_pivot = self.relative_rotation_pivot
        current_pivot = tip_positions[1]
        fingertip_velocities = {}
        for finger in expected_fingers:
            indices = np.asarray(FINGER_JOINT_INDEX[finger], dtype=int)
            fingertip_velocities[finger] = (
                tip_jacobian(
                    self.hand_q,
                    finger,
                    eps=self.cfg.jacobian_eps,
                )
                @ qdot[indices]
            )
        thumb_velocity = fingertip_velocities[1]
        driven_fingers = [
            finger
            for finger in self.relative_rotation_start_fingertips
            if finger != 1
        ]
        if not driven_fingers:
            self.relative_rotation_phase = "rotation_error"
            self._log("[RELATIVE_ROTATION] stopped: no non-thumb driven finger")
            return rotation_forces
        angular_numerator = 0.0
        angular_denominator = 0.0
        for finger in driven_fingers:
            radius = tip_positions[finger] - current_pivot
            planar_radius = radius - np.dot(radius, axis) * axis
            relative_velocity = fingertip_velocities[finger] - thumb_velocity
            angular_numerator += float(
                np.dot(axis, np.cross(planar_radius, relative_velocity))
            )
            angular_denominator += float(np.dot(planar_radius, planar_radius))
        measured_angular_velocity = (
            angular_numerator / angular_denominator
            if angular_denominator > 1e-12
            else 0.0
        )
        velocity_alpha = float(
            np.clip(self.cfg.relative_rotation_velocity_alpha, 0.0, 1.0)
        )
        self.relative_rotation_angular_velocity = (
            (1.0 - velocity_alpha) * self.relative_rotation_angular_velocity
            + velocity_alpha * measured_angular_velocity
        )
        axis_cross = np.array(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ],
            dtype=np.float64,
        )
        reference_rotation = (
            np.eye(3, dtype=np.float64)
            + np.sin(reference_angle) * axis_cross
            + (1.0 - np.cos(reference_angle)) * (axis_cross @ axis_cross)
        )
        target_angle = self.relative_rotation_target_rad
        target_rotation = (
            np.eye(3, dtype=np.float64)
            + np.sin(target_angle) * axis_cross
            + (1.0 - np.cos(target_angle)) * (axis_cross @ axis_cross)
        )

        final_errors = {}
        for finger in driven_fingers:
            start_position = self.relative_rotation_start_fingertips[finger]
            start_radius = start_position - start_pivot
            final_target = current_pivot + target_rotation @ start_radius
            final_errors[finger] = final_target - tip_positions[finger]
        maximum_final_error = max(
            (float(np.linalg.norm(error)) for error in final_errors.values()),
            default=float("inf"),
        )

        if (
            self.relative_rotation_phase != "rotation_reached"
            and reference_progress >= 1.0 - 1e-12
            and maximum_final_error <= position_tolerance
        ):
            self.relative_rotation_phase = "rotation_reached"
            self._log(
                "[RELATIVE_ROTATION] target reached; thumb-pivot hold active "
                f"max_tip_error_mm={1000.0 * maximum_final_error:.3f}, "
                f"current_deg="
                f"{np.degrees(self.relative_rotation_current_rad):.3f}"
            )

        # Thumb receives only the ordinary grasp force. Every other active
        # finger tracks a target rotated about the command-time thumb contact.
        rotation_forces[1][:] = 0.0
        for finger in driven_fingers:
            start_position = self.relative_rotation_start_fingertips[finger]
            start_radius = start_position - start_pivot
            rho = float(np.linalg.norm(start_radius))
            target_position = current_pivot + reference_rotation @ start_radius
            target_velocity = (
                thumb_velocity
                + reference_angular_velocity
                * np.cross(axis, reference_rotation @ start_radius)
            )
            position_error = target_position - tip_positions[finger]
            error_norm = float(np.linalg.norm(position_error))
            if error_norm > position_error_limit:
                position_error *= position_error_limit / error_norm

            fingertip_velocity = fingertip_velocities[finger]
            velocity_error = target_velocity - fingertip_velocity
            force = (
                position_kp * position_error
                + position_kd * velocity_error
            ) / max(rho, radius_min)
            force_norm = float(np.linalg.norm(force))
            if force_limit > 0.0 and force_norm > force_limit:
                force *= force_limit / force_norm
            rotation_forces[finger] = force

        self.relative_rotation_command_moment = sum(
            (
                float(
                    np.dot(
                        np.cross(
                            position - current_pivot,
                            rotation_forces[finger],
                        ),
                        axis,
                    )
                )
                for finger, position in tip_positions.items()
            ),
            0.0,
        )

        return rotation_forces

    def _calc_relative_translation_forces(
        self,
        cg: np.ndarray,
        tip_positions: Dict[int, np.ndarray],
        now: float,
        qdot: np.ndarray | None = None,
    ) -> Dict[int, np.ndarray]:
        """Calculate per-fingertip Cartesian PD forces for translation."""

        zero = self._zero_translation_forces(tip_positions)
        self.relative_translation_command_force[:] = 0.0
        self.relative_translation_shape_forces = _copy_finger_vectors(zero)
        self.relative_translation_control_axis_drive_force = 0.0
        active_phases = {"translating", "translation_reached"}
        if self.relative_translation_phase not in active_phases:
            return zero

        cg = np.asarray(cg, dtype=np.float64)
        tip_positions = {
            int(finger): np.asarray(position, dtype=np.float64)
            for finger, position in tip_positions.items()
        }
        qdot = (
            np.zeros(JOINT_COUNT, dtype=np.float64)
            if qdot is None
            else np.asarray(qdot, dtype=np.float64)
        )
        now = float(now)
        expected_fingers = set(self.relative_translation_target_fingertips)
        if (
            cg.shape != (3,)
            or not np.all(np.isfinite(cg))
            or qdot.shape != (JOINT_COUNT,)
            or not np.all(np.isfinite(qdot))
            or not np.isfinite(now)
            or set(tip_positions) != expected_fingers
            or set(self.relative_translation_contact_weights) != expected_fingers
            or any(
                position.shape != (3,) or not np.all(np.isfinite(position))
                for position in tip_positions.values()
            )
        ):
            self.relative_translation_phase = "translation_error"
            self._log(
                "[RELATIVE_TRANSLATION] stopped: invalid centroid/contact geometry/time"
            )
            return zero

        velocity_alpha = float(
            np.clip(self.cfg.relative_translation_velocity_alpha, 0.0, 1.0)
        )
        measured_fingertip_velocities = {}
        for finger in tip_positions:
            indices = np.asarray(FINGER_JOINT_INDEX[finger], dtype=int)
            measured_velocity = (
                tip_jacobian(
                    self.hand_q,
                    finger,
                    eps=self.cfg.jacobian_eps,
                )
                @ qdot[indices]
            )
            measured_fingertip_velocities[finger] = measured_velocity
            previous_velocity = self.relative_translation_fingertip_velocities.get(
                finger,
                np.zeros(3, dtype=np.float64),
            )
            self.relative_translation_fingertip_velocities[finger] = (
                (1.0 - velocity_alpha) * previous_velocity
                + velocity_alpha * measured_velocity
            )
        measured_centroid_velocity = sum(
            (
                float(self.relative_translation_contact_weights[finger])
                * velocity
                for finger, velocity in measured_fingertip_velocities.items()
            ),
            np.zeros(3, dtype=np.float64),
        )
        self.relative_translation_centroid_velocity = (
            (1.0 - velocity_alpha) * self.relative_translation_centroid_velocity
            + velocity_alpha * measured_centroid_velocity
        )
        elapsed = (
            0.0
            if self.relative_translation_started_at is None
            else max(0.0, now - float(self.relative_translation_started_at))
        )
        ramp_sec = max(
            0.0,
            float(self.cfg.relative_translation_reference_ramp_sec),
        )
        if not np.isfinite(ramp_sec):
            self.relative_translation_phase = "translation_error"
            self._log("[RELATIVE_TRANSLATION] stopped: invalid reference ramp")
            return zero
        ramp_linear = 1.0 if ramp_sec <= 0.0 else np.clip(elapsed / ramp_sec, 0.0, 1.0)
        reference_progress = float(
            ramp_linear * ramp_linear * (3.0 - 2.0 * ramp_linear)
        )
        self.relative_translation_reference_progress = reference_progress

        reference_centroid = (
            self.relative_translation_start_centroid
            + reference_progress * self.relative_translation_delta
        )
        control_centroid_error = reference_centroid - cg
        self.relative_translation_error = (
            self.relative_translation_target_centroid - cg
        )
        fingertip_errors = {
            finger: self.relative_translation_target_fingertips[finger] - position
            for finger, position in tip_positions.items()
        }
        control_fingertip_errors = {
            finger: (
                self.relative_translation_start_fingertips[finger]
                + reference_progress * self.relative_translation_delta
                - position
            )
            for finger, position in tip_positions.items()
        }
        translation_distance = float(np.linalg.norm(self.relative_translation_delta))
        if translation_distance <= 1e-12:
            self.relative_translation_phase = "translation_error"
            self._log("[RELATIVE_TRANSLATION] stopped: zero translation direction")
            return zero
        translation_axis = self.relative_translation_delta / translation_distance
        self.relative_translation_max_axis_fingertip_error = max(
            float(np.linalg.norm(error))
            for error in fingertip_errors.values()
        )
        self.relative_translation_control_axis_error = float(
            np.dot(translation_axis, control_centroid_error)
        )

        timeout = float(self.cfg.relative_translation_timeout_sec)
        if (
            self.relative_translation_phase != "translation_reached"
            and np.isfinite(timeout)
            and timeout > 0.0
            and elapsed >= timeout
        ):
            self.relative_translation_phase = "translation_timeout"
            self.relative_translation_reached_since = None
            self._log(
                "[RELATIVE_TRANSLATION] timeout; position control removed "
                f"centroid_error_mm="
                f"{np.round(1000.0 * self.relative_translation_error, 3).tolist()}, "
                f"max_tip_error_mm="
                f"{1000.0 * self.relative_translation_max_axis_fingertip_error:.3f}"
            )
            return zero

        position_tolerance = max(
            0.0,
            float(self.cfg.relative_translation_position_tolerance_m),
        )
        velocity_tolerance = max(
            0.0,
            float(self.cfg.relative_translation_velocity_tolerance_mps),
        )
        axis_position_error = abs(
            float(np.dot(translation_axis, self.relative_translation_error))
        )
        maximum_fingertip_velocity = max(
            (float(np.linalg.norm(velocity))
             for velocity in self.relative_translation_fingertip_velocities.values()),
            default=0.0,
        )
        translation_velocity = max(
            float(np.linalg.norm(self.relative_translation_centroid_velocity)),
            maximum_fingertip_velocity,
        )
        inside_tolerance = (
            reference_progress >= 1.0 - 1e-12
            and axis_position_error <= position_tolerance
            and self.relative_translation_max_axis_fingertip_error
            <= position_tolerance
            and translation_velocity <= velocity_tolerance
        )
        if inside_tolerance:
            if self.relative_translation_reached_since is None:
                self.relative_translation_reached_since = now
            settle_sec = max(
                0.0,
                float(self.cfg.relative_translation_settle_sec),
            )
            if (
                self.relative_translation_phase != "translation_reached"
                and now - self.relative_translation_reached_since >= settle_sec
            ):
                self.relative_translation_phase = "translation_reached"
                self._log(
                    "[RELATIVE_TRANSLATION] target reached "
                    f"centroid_error_mm="
                    f"{np.round(1000.0 * self.relative_translation_error, 3).tolist()}, "
                    f"max_tip_error_mm="
                    f"{1000.0 * self.relative_translation_max_axis_fingertip_error:.3f}"
                )
        else:
            self.relative_translation_reached_since = None
            if (
                self.relative_translation_phase == "translation_reached"
                and max(
                    axis_position_error,
                    self.relative_translation_max_axis_fingertip_error,
                )
                > 1.5 * max(position_tolerance, 1e-9)
            ):
                self.relative_translation_phase = "translating"

        kp = float(self.cfg.relative_translation_kp)
        kd = float(self.cfg.relative_translation_kd)
        gains = (kp, kd)
        if not all(np.isfinite(gain) and gain >= 0.0 for gain in gains):
            self.relative_translation_phase = "translation_error"
            self._log("[RELATIVE_TRANSLATION] stopped: invalid Cartesian gains")
            return zero

        reference_velocity = np.zeros(3, dtype=np.float64)
        if 0.0 < ramp_linear < 1.0 and ramp_sec > 0.0:
            reference_velocity = (
                6.0 * ramp_linear * (1.0 - ramp_linear) / ramp_sec
            ) * self.relative_translation_delta

        contact_scale = 1.0 / len(control_fingertip_errors)
        forces = {
            finger: contact_scale * (kp * error + kd * (
                reference_velocity
                - self.relative_translation_fingertip_velocities[finger]
            ))
            for finger, error in control_fingertip_errors.items()
        }
        self.relative_translation_control_axis_drive_force = float(
            np.dot(translation_axis, sum(forces.values(), np.zeros(3)))
        )
        for force in forces.values():
            if force.shape != (3,) or not np.all(np.isfinite(force)):
                self.relative_translation_phase = "translation_error"
                self._log("[RELATIVE_TRANSLATION] stopped: non-finite task force")
                return zero

        resultant = sum(forces.values(), np.zeros(3, dtype=np.float64))
        force_limit = max(
            0.0,
            float(self.cfg.relative_translation_force_limit),
        )
        scale = 1.0
        resultant_norm = float(np.linalg.norm(resultant))
        if force_limit > 0.0 and resultant_norm > force_limit:
            scale = min(scale, force_limit / resultant_norm)

        per_finger_limit = max(
            0.0,
            float(self.cfg.relative_translation_per_finger_force_limit),
        )
        if per_finger_limit > 0.0:
            maximum_contact_force = max(
                float(np.linalg.norm(force))
                for force in forces.values()
            )
            if maximum_contact_force > per_finger_limit:
                scale = min(scale, per_finger_limit / maximum_contact_force)

        if scale < 1.0:
            for finger in forces:
                forces[finger] *= scale
        self.relative_translation_command_force = sum(
            forces.values(),
            np.zeros(3, dtype=np.float64),
        )
        return forces

    def _clip_regular_grasp_tau(self, tau: np.ndarray) -> np.ndarray:
        """Apply the same joint limits used by the Cartesian grasp policy."""

        clipped = np.clip(
            np.asarray(tau, dtype=np.float64),
            -float(self.cfg.groped_tau_limit),
            float(self.cfg.groped_tau_limit),
        )
        for joint_index, limit in BASE_JOINT_TAU_LIMIT.items():
            clipped[joint_index] = np.clip(
                clipped[joint_index], -float(limit), float(limit)
            )
        return clipped


    def _set_card_phase(
        self,
        phase: str,
        now: float,
        tip_positions: Dict[int, np.ndarray] | None = None,
    ) -> None:
        self.card_phase = str(phase)
        self.card_phase_started_at = float(now)
        self.card_stall_reference_positions = {
            int(finger): np.asarray(position, dtype=np.float64).copy()
            for finger, position in (tip_positions or {}).items()
        }
        self.card_stall_since = float(now)
        self.card_stall_max_motion_m = 0.0
        self.card_stable_elapsed_sec = 0.0
        self.card_floor_stall_since = {
            finger: float(now)
            for finger in self.card_stall_reference_positions
        }
        self.card_floor_stable_elapsed_sec = {
            finger: 0.0
            for finger in self.card_stall_reference_positions
        }
        self.card_floor_motion_m = {
            finger: 0.0
            for finger in self.card_stall_reference_positions
        }
        self.card_floor_contact_detected = {
            finger: False
            for finger in self.card_stall_reference_positions
        }
        self.card_index_tip_reached_since = None
        self.card_thumb_j1_hold_error_rad = 0.0
        self.card_thumb_j1_hold_tau = 0.0

    def _card_floor_contacts_ready(
        self,
        tip_positions: Dict[int, np.ndarray],
        now: float,
    ) -> bool:
        threshold = float(self.cfg.card_tip_stall_threshold_m)
        duration = float(self.cfg.card_floor_stall_sec)
        for finger, position in tip_positions.items():
            if self.card_floor_contact_detected.get(finger, False):
                start = self.card_floor_stall_since[finger]
                self.card_floor_stable_elapsed_sec[finger] = max(
                    0.0,
                    float(now) - float(start),
                )
                continue

            reference = self.card_stall_reference_positions.get(finger)
            if reference is None:
                self.card_stall_reference_positions[finger] = np.asarray(
                    position,
                    dtype=np.float64,
                ).copy()
                self.card_floor_stall_since[finger] = float(now)
                self.card_floor_stable_elapsed_sec[finger] = 0.0
                self.card_floor_motion_m[finger] = 0.0
                self.card_floor_contact_detected[finger] = False
                continue

            motion = float(
                np.linalg.norm(
                    np.asarray(position, dtype=np.float64) - reference
                )
            )
            self.card_floor_motion_m[finger] = motion
            if motion > threshold:
                self.card_stall_reference_positions[finger] = np.asarray(
                    position,
                    dtype=np.float64,
                ).copy()
                self.card_floor_stall_since[finger] = float(now)
                self.card_floor_stable_elapsed_sec[finger] = 0.0
                continue

            stable_elapsed = max(
                0.0,
                float(now) - float(self.card_floor_stall_since[finger]),
            )
            self.card_floor_stable_elapsed_sec[finger] = stable_elapsed
            if stable_elapsed >= duration:
                self.card_floor_contact_detected[finger] = True
                self.card_floor_contact_positions[finger] = np.asarray(
                    position,
                    dtype=np.float64,
                ).copy()
                name = "thumb" if finger == CARD_THUMB_ID else "index"
                if finger == CARD_THUMB_ID:
                    thumb_j1 = FINGER_JOINT_INDEX[CARD_THUMB_ID][0]
                    self.card_thumb_j1_hold_target = float(
                        self.hand_q[thumb_j1]
                    )
                self._log(
                    f"[CARD] {name} floor contact detected; "
                    f"stable_for_sec={stable_elapsed:.3f}, "
                    f"stall_motion_mm={1000.0 * motion:.3f}"
                    + (
                        ", thumb_j1_hold_target_rad="
                        f"{self.card_thumb_j1_hold_target:.4f}"
                        if finger == CARD_THUMB_ID
                        else ""
                    )
                )

        self.card_stall_max_motion_m = max(
            self.card_floor_motion_m.values(),
            default=0.0,
        )
        self.card_stable_elapsed_sec = min(
            self.card_floor_stable_elapsed_sec.values(),
            default=0.0,
        )
        return bool(self.card_floor_contact_detected) and all(
            self.card_floor_contact_detected.get(finger, False)
            for finger in tip_positions
        )

    def _card_tips_stalled(
        self,
        tip_positions: Dict[int, np.ndarray],
        now: float,
        duration: float,
    ) -> bool:
        threshold = float(self.cfg.card_tip_stall_threshold_m)
        if set(self.card_stall_reference_positions) != set(tip_positions):
            self._set_card_phase(self.card_phase, now, tip_positions)
            return False

        motions = [
            float(
                np.linalg.norm(
                    np.asarray(position, dtype=np.float64)
                    - self.card_stall_reference_positions[finger]
                )
            )
            for finger, position in tip_positions.items()
        ]
        self.card_stall_max_motion_m = max(motions, default=0.0)
        moved = self.card_stall_max_motion_m > threshold
        if moved:
            self.card_stall_reference_positions = _copy_finger_vectors(
                tip_positions
            )
            self.card_stall_since = float(now)
            self.card_stable_elapsed_sec = 0.0
            return False
        self.card_stable_elapsed_sec = (
            0.0
            if self.card_stall_since is None
            else max(0.0, float(now) - float(self.card_stall_since))
        )
        return self.card_stable_elapsed_sec >= float(duration)

    def _card_force_result(
        self,
        tip_positions: Dict[int, np.ndarray],
        forces: Dict[int, np.ndarray],
    ) -> GraspPolicyResult:
        forces = _copy_finger_vectors(forces)
        tip_positions = _copy_finger_vectors(tip_positions)
        zero_forces = {
            finger: np.zeros(3, dtype=np.float64)
            for finger in forces
        }
        points = np.stack(list(tip_positions.values()))
        cg = np.mean(points, axis=0)
        return GraspPolicyResult(
            tau=self.policy.calc_tau_from_total_forces(self.hand_q, forces),
            alpha={
                finger: float(np.linalg.norm(force))
                for finger, force in forces.items()
            },
            cg=cg.copy(),
            cv=cg.copy(),
            fingertip_positions=tip_positions,
            grasp_forces=_copy_finger_vectors(forces),
            rotation_forces=_copy_finger_vectors(zero_forces),
            center_hold_forces=_copy_finger_vectors(zero_forces),
            collision_forces=_copy_finger_vectors(zero_forces),
            total_forces=_copy_finger_vectors(forces),
        )

    def _abort_card_grasp(self, now: float, reason: str) -> None:
        self._log(f"[CARD] stopped: {reason}")
        self.cancel_card_grasp()
        self.active_finger_count = 0
        self.state = "PRE_GRASP_POSE"
        self.state_start = float(now)
        self._reset_inactive_collision_avoidance()

    def _start_card_grasp(self, now: float) -> bool:
        if self.state != "PRE_GRASP_POSE" or self.pose_type != 4:
            self._log(
                "[CARD] ignored: grasp_type=7 requires CARD_PRE_GRASP_POSE "
                f"(state={self.state}, pose_type={self.pose_type})"
            )
            return False

        values = (
            self.cfg.card_floor_force_n,
            self.cfg.card_floor_hold_force_n,
            self.cfg.card_pinch_force_n,
            self.cfg.card_index_tip_target_deg,
            self.cfg.card_index_tip_kp,
            self.cfg.card_index_tip_kd,
            self.cfg.card_index_tip_tau_limit,
            self.cfg.card_index_tip_tolerance_deg,
            self.cfg.card_index_tip_stable_sec,
            self.cfg.card_tip_stall_threshold_m,
            self.cfg.card_floor_stall_sec,
            self.cfg.card_pinch_stall_sec,
            self.cfg.card_post_pinch_delay_sec,
            self.cfg.card_thumb_j1_hold_kp,
            self.cfg.card_thumb_j1_hold_kd,
            self.cfg.card_thumb_j1_hold_tau_limit,
            self.cfg.card_floor_timeout_sec,
            self.cfg.card_pinch_timeout_sec,
            self.cfg.card_joint_state_timeout_sec,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in values):
            self._log("[CARD] ignored: every CARD gain/limit must be finite and > 0")
            return False
        if (
            self.cfg.card_index_tip_target_deg >= 180.0
            or not np.isfinite(self.cfg.card_index_tip_return_deg)
            or abs(self.cfg.card_index_tip_return_deg) >= 180.0
        ):
            self._log("[CARD] ignored: index tip targets must be within 180 deg")
            return False
        self.cancel_relative_rotation()
        self.cancel_relative_translation()
        self.cancel_card_grasp()
        self._reset_regular_force_balance_state()
        self._reset_envelop_grasp()
        self._reset_inactive_collision_avoidance()
        self._clear_transition_state()

        self.use_fingers = [CARD_THUMB_ID, CARD_INDEX_ID]
        self.policy = GraspPolicy(self.use_fingers, self.cfg)
        self.active_finger_count = 7
        self.state = "CARD_GRASP"
        self.state_start = float(now)
        tips = {
            finger: tip_position(self.hand_q, finger)
            for finger in self.use_fingers
        }
        self.card_floor_contact_positions = _copy_finger_vectors(tips)
        thumb_j1 = FINGER_JOINT_INDEX[CARD_THUMB_ID][0]
        self.card_thumb_j1_hold_target = float(self.hand_q[thumb_j1])
        self._set_card_phase(
            "card_pinch",
            now,
            {CARD_INDEX_ID: tips[CARD_INDEX_ID]},
        )
        self._log(
            "[COMMAND] grasp_type=7 -> CARD_GRASP, phase=card_pinch"
        )
        return True

    def _card_thumb_j1_hold_pd(self, qdot: np.ndarray) -> tuple[float, float]:
        if self.card_thumb_j1_hold_target is None:
            return 0.0, 0.0
        thumb_j1 = FINGER_JOINT_INDEX[CARD_THUMB_ID][0]
        tau, error = pose_pd(
            [self.card_thumb_j1_hold_target],
            [self.hand_q[thumb_j1]],
            [qdot[thumb_j1]],
            kp=self.cfg.card_thumb_j1_hold_kp,
            kd=self.cfg.card_thumb_j1_hold_kd,
            limit=self.cfg.card_thumb_j1_hold_tau_limit,
        )
        self.card_thumb_j1_hold_error_rad = float(error[0])
        self.card_thumb_j1_hold_tau = float(tau[0])
        return float(tau[0]), float(error[0])

    def _calc_card_grasp(
        self,
        qdot: np.ndarray,
        now: float,
    ):
        qdot = np.asarray(qdot, dtype=np.float64)
        tip_fingers = [CARD_THUMB_ID, CARD_INDEX_ID]
        tip_positions = {
            finger: tip_position(self.hand_q, finger)
            for finger in tip_fingers
        }
        phase_started_at = self.card_phase_started_at
        if phase_started_at is None:
            self._abort_card_grasp(now, "missing phase start time")
            zero = {finger: np.zeros(3) for finger in self.use_fingers}
            return self._card_force_result(tip_positions, zero), np.zeros(
                JOINT_COUNT
            ), np.zeros(JOINT_COUNT)

        if self.card_phase not in {
            "card_floor_contact",
            "card_pinch",
            "card_post_pinch_wait",
            "card_index_tip_flex",
            "card_index_tip_return",
        }:
            self._abort_card_grasp(now, f"unknown phase {self.card_phase}")
            zero = {finger: np.zeros(3) for finger in self.use_fingers}
            return self._card_force_result(tip_positions, zero), np.zeros(
                JOINT_COUNT
            ), np.zeros(JOINT_COUNT)

        timeout = {
            "card_floor_contact": float(self.cfg.card_floor_timeout_sec),
            "card_pinch": float(self.cfg.card_pinch_timeout_sec),
        }.get(self.card_phase)
        if (
            timeout is not None
            and float(now) - float(phase_started_at) >= timeout
        ):
            phase = self.card_phase
            if phase == "card_floor_contact":
                motion_mm = {
                    finger: round(1000.0 * value, 3)
                    for finger, value in self.card_floor_motion_m.items()
                }
                detail = (
                    "contact="
                    f"{self.card_floor_contact_detected}, "
                    "stable_for_sec="
                    f"{self.card_floor_stable_elapsed_sec}, "
                    "stall_motion_mm="
                    f"{motion_mm}, "
                    "threshold_mm="
                    f"{1000.0 * float(self.cfg.card_tip_stall_threshold_m):.3f}"
                )
            else:
                detail = (
                    f"stable_for_sec={self.card_stable_elapsed_sec:.3f}, "
                    f"stall_motion_mm={1000.0 * self.card_stall_max_motion_m:.3f}, "
                    "threshold_mm="
                    f"{1000.0 * float(self.cfg.card_tip_stall_threshold_m):.3f}"
                )
            self._abort_card_grasp(now, f"{phase} timeout; {detail}")
            zero = {finger: np.zeros(3) for finger in self.use_fingers}
            return self._card_force_result(tip_positions, zero), np.zeros(
                JOINT_COUNT
            ), np.zeros(JOINT_COUNT)

        world_down_in_hand = self.rotation_hand_to_world.T @ np.array(
            [0.0, 0.0, -1.0],
            dtype=np.float64,
        )
        if self.card_phase == "card_floor_contact":
            forces = {
                finger: float(self.cfg.card_floor_force_n) * world_down_in_hand
                for finger in (CARD_THUMB_ID, CARD_INDEX_ID)
            }
            if self._card_floor_contacts_ready(tip_positions, now):
                stable_for = dict(self.card_floor_stable_elapsed_sec)
                motion_mm = {
                    finger: round(1000.0 * motion, 3)
                    for finger, motion in self.card_floor_motion_m.items()
                }
                self._set_card_phase(
                    "card_pinch",
                    now,
                    {CARD_INDEX_ID: tip_positions[CARD_INDEX_ID]},
                )
                self._log(
                    "[CARD] both floor contacts ready; "
                    f"stable_for_sec={stable_for}, "
                    f"stall_motion_mm={motion_mm}; phase=card_pinch"
                )

        else:  # card pinch, post-pinch wait, index flex, or index return
            if set(self.card_floor_contact_positions) != {
                CARD_THUMB_ID,
                CARD_INDEX_ID,
            }:
                self._abort_card_grasp(now, "missing floor contact positions")
                zero = {finger: np.zeros(3) for finger in self.use_fingers}
                return self._card_force_result(tip_positions, zero), np.zeros(
                    JOINT_COUNT
                ), np.zeros(JOINT_COUNT)
            down_force = (
                float(self.cfg.card_floor_hold_force_n) * world_down_in_hand
            )
            toward_thumb = (
                tip_positions[CARD_THUMB_ID]
                - tip_positions[CARD_INDEX_ID]
            )
            distance = float(np.linalg.norm(toward_thumb))
            if distance <= 1e-9:
                self._abort_card_grasp(now, "thumb/index direction is degenerate")
                zero = {finger: np.zeros(3) for finger in self.use_fingers}
                return self._card_force_result(tip_positions, zero), np.zeros(
                    JOINT_COUNT
                ), np.zeros(JOINT_COUNT)
            toward_thumb /= distance
            toward_thumb_world = self.rotation_hand_to_world @ toward_thumb
            toward_thumb_world[2] = 0.0
            horizontal_distance = float(np.linalg.norm(toward_thumb_world))
            if horizontal_distance <= 1e-9:
                self._abort_card_grasp(
                    now,
                    "thumb/index horizontal direction is degenerate",
                )
                zero = {finger: np.zeros(3) for finger in self.use_fingers}
                return self._card_force_result(tip_positions, zero), np.zeros(
                    JOINT_COUNT
                ), np.zeros(JOINT_COUNT)
            toward_thumb = (
                self.rotation_hand_to_world.T
                @ (toward_thumb_world / horizontal_distance)
            )
            forces = {
                CARD_THUMB_ID: (
                    down_force
                    if self.card_phase in {
                        "card_index_tip_flex",
                        "card_index_tip_return",
                    }
                    else np.zeros(3, dtype=np.float64)
                ),
                CARD_INDEX_ID: down_force
                + float(self.cfg.card_pinch_force_n) * toward_thumb,
            }
            if (
                self.card_phase == "card_pinch"
                and self._card_tips_stalled(
                    {CARD_INDEX_ID: tip_positions[CARD_INDEX_ID]},
                    now,
                    self.cfg.card_pinch_stall_sec,
                )
            ):
                self._log(
                    "[CARD] pinch contact detected; "
                    f"stable_for_sec={self.card_stable_elapsed_sec:.3f}, "
                    "stall_motion_mm="
                    f"{1000.0 * self.card_stall_max_motion_m:.3f}; "
                    "phase=card_post_pinch_wait"
                )
                index_hold_joints = np.asarray(
                    FINGER_JOINT_INDEX[CARD_INDEX_ID][:-1],
                    dtype=int,
                )
                self.card_index_flex_hold_target = self.hand_q[
                    index_hold_joints
                ].copy()
                self.card_floor_contact_positions = _copy_finger_vectors(
                    tip_positions
                )
                self._set_card_phase("card_post_pinch_wait", now)
            if (
                self.card_phase == "card_post_pinch_wait"
                and float(now) - float(self.card_phase_started_at)
                >= float(self.cfg.card_post_pinch_delay_sec)
            ):
                self._log(
                    "[CARD] post-pinch wait complete; "
                    f"delay_sec={self.cfg.card_post_pinch_delay_sec:.3f}; "
                    "phase=card_index_tip_flex"
                )
                self._set_card_phase("card_index_tip_flex", now)
            if self.card_phase in {
                "card_index_tip_flex",
                "card_index_tip_return",
            }:
                index_tip_joint = FINGER_JOINT_INDEX[CARD_INDEX_ID][-1]
                index_target_deg = (
                    float(self.cfg.card_index_tip_target_deg)
                    if self.card_phase == "card_index_tip_flex"
                    else float(self.cfg.card_index_tip_return_deg)
                )
                index_tip_error = float(
                    np.deg2rad(index_target_deg)
                    - self.hand_q[index_tip_joint]
                )
                tolerance = np.deg2rad(
                    float(self.cfg.card_index_tip_tolerance_deg)
                )
                if (
                    self.card_phase == "card_index_tip_return"
                    and abs(index_tip_error) <= tolerance
                ):
                    self._log(
                        "[CARD] index J4 return target reached; "
                        f"target_deg={index_target_deg:.3f}; "
                        f"error_deg={np.degrees(index_tip_error):.3f}; "
                        "switching to grasp_type=1"
                    )
                    self.apply_grasp_type(1, now, internal=True)
                elif abs(index_tip_error) <= tolerance:
                    if self.card_index_tip_reached_since is None:
                        self.card_index_tip_reached_since = float(now)
                    elif (
                        float(now) - float(self.card_index_tip_reached_since)
                        >= float(self.cfg.card_index_tip_stable_sec)
                    ):
                        self._log(
                            "[CARD] index J4 lift target reached; "
                            f"error_deg={np.degrees(index_tip_error):.3f}; "
                            "return_target_deg="
                            f"{self.cfg.card_index_tip_return_deg:.3f}; "
                            "phase=card_index_tip_return"
                        )
                        self._set_card_phase("card_index_tip_return", now)
                else:
                    self.card_index_tip_reached_since = None

        result = self._card_force_result(tip_positions, forces)
        inactive_pd = self._inactive_pre_grasp_pd(
            [CARD_THUMB_ID, CARD_INDEX_ID],
            qdot,
            collision_avoidance_enabled=False,
        )
        err = np.zeros(JOINT_COUNT, dtype=np.float64)
        if self.card_phase in {"card_index_tip_flex", "card_index_tip_return"}:
            index_joints = np.asarray(
                FINGER_JOINT_INDEX[CARD_INDEX_ID],
                dtype=int,
            )
            index_hold_joints = index_joints[:-1]
            if self.card_index_flex_hold_target is None:
                self._abort_card_grasp(now, "missing index flex hold target")
                return result, np.zeros(JOINT_COUNT), err
            hold_tau, hold_error = pose_pd(
                self.card_index_flex_hold_target,
                self.hand_q[index_hold_joints],
                qdot[index_hold_joints],
                kp=self.cfg.pose_kp,
                kd=self.cfg.pose_kd,
                limit=self.cfg.pose_pd_limit,
            )
            result.tau[index_hold_joints] += hold_tau
            err[index_hold_joints] = hold_error
            index_tip_joint = index_joints[-1]
            target_deg = (
                float(self.cfg.card_index_tip_target_deg)
                if self.card_phase == "card_index_tip_flex"
                else float(self.cfg.card_index_tip_return_deg)
            )
            tip_tau, tip_error = pose_pd(
                [np.deg2rad(target_deg)],
                [self.hand_q[index_tip_joint]],
                [qdot[index_tip_joint]],
                kp=self.cfg.card_index_tip_kp,
                kd=self.cfg.card_index_tip_kd,
                limit=self.cfg.card_index_tip_tau_limit,
            )
            result.tau[index_tip_joint] += tip_tau[0]
            result.tau = self._clip_regular_grasp_tau(result.tau)
            err[index_tip_joint] = tip_error[0]
        thumb_j1 = FINGER_JOINT_INDEX[CARD_THUMB_ID][0]
        hold_tau, hold_error = self._card_thumb_j1_hold_pd(qdot)
        inactive_pd[thumb_j1] += hold_tau
        err[thumb_j1] = hold_error
        return result, inactive_pd, err

    def sync_joint_state(self, q: np.ndarray) -> None:
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (JOINT_COUNT,):
            raise ValueError(f"q must have shape ({JOINT_COUNT},)")
        self.hand_q = q.copy()

    def _current_pre_grasp_pose(self):
        return self.pose_type_targets.get(
            self.pre_grasp_pose_type,
            self.pose_type_targets[2],
        )

    def _pose_target_for_type(self, pose_type):
        return self.pose_type_targets.get(pose_type, self.pose_type_targets[1])

    def _apply_pose_type_command(self, pose_type, now):
        self.cancel_relative_rotation()
        self.cancel_relative_translation()
        self.cancel_card_grasp()
        self._reset_regular_force_balance_state()
        self.active_finger_count = 0
        self._clear_transition_state()
        self._reset_envelop_grasp()

        self.pose_type = pose_type
        if pose_type == 1:
            self.pre_grasp_pose_type = 2
            state = "NORMAL_POSE"
            label = "normal pose"
        else:
            self.pre_grasp_pose_type = pose_type
            state = "PRE_GRASP_POSE"
            label = {
                2: "default pre-grasp",
                3: "compact pre-grasp",
                4: "card pre-grasp",
                5: "rotation pre-grasp",
                6: "rotation pre-grasp (blind grasping)",
            }[pose_type]

        self._log(f"[POSE_TYPE] pose_type={pose_type} -> {label}")
        return state, now, 0.0

    def _reset_inactive_collision_avoidance(self):
        self.inactive_collision_avoidance_offsets_rad[:] = 0.0
        self.inactive_collision_avoidance_active = [False] * 5
        self.inactive_collision_follow_source[:] = 0
        self.inactive_collision_follow_offset_rad[:] = 0.0
        self.inactive_collision_approach_direction[:] = 0.0
        self.inactive_collision_previous_joint_positions = None
        self.inactive_collision_min_clearance_m = -1.0

    def _update_inactive_collision_avoidance(self, active_fingers, qdot):
        """Make an inactive finger follow a colliding adjacent joint."""

        # ponytail: joint-only heuristic; restore geometry checks if
        # unmonitored links or grasped-object collisions become relevant.
        if not bool(self.cfg.inactive_collision_avoidance_enable):
            self._reset_inactive_collision_avoidance()
            return self.inactive_collision_avoidance_offsets_rad.copy()

        active_set = {int(finger) for finger in active_fingers}
        qdot = np.asarray(qdot, dtype=np.float64)
        tolerance = max(
            0.0,
            float(self.cfg.inactive_collision_joint_match_tolerance_rad),
        )
        release_margin = max(
            0.0,
            float(self.cfg.inactive_collision_joint_release_margin_rad),
        )
        pre_grasp_pose = self._current_pre_grasp_pose()
        joint_indices = {
            finger: int(
                FINGER_JOINT_INDEX[finger][
                    FINGER_AVOIDANCE_JOINT_LOCAL_INDEX[finger]
                ]
            )
            for finger in INACTIVE_COLLISION_CHAIN
        }
        current = {
            finger: float(self.hand_q[joint_indices[finger]])
            for finger in INACTIVE_COLLISION_CHAIN
        }
        previous = self.inactive_collision_previous_joint_positions

        for finger in INACTIVE_COLLISION_CHAIN:
            state_index = finger - 1
            if finger in active_set:
                self.inactive_collision_follow_source[state_index] = 0
                self.inactive_collision_follow_offset_rad[state_index] = 0.0
                self.inactive_collision_approach_direction[state_index] = 0.0

        # Release as soon as the source is safely away. The ordinary inactive-
        # finger PD completes any remaining return to the waiting pose.
        for finger in INACTIVE_COLLISION_CHAIN:
            state_index = finger - 1
            source = int(self.inactive_collision_follow_source[state_index])
            if source == 0:
                continue
            waiting = float(pre_grasp_pose[joint_indices[finger]])
            direction = float(
                self.inactive_collision_approach_direction[state_index]
            )
            source_target = (
                current[source]
                + self.inactive_collision_follow_offset_rad[state_index]
            )
            source_idle = (
                source not in active_set
                and self.inactive_collision_follow_source[source - 1] == 0
            )
            if (
                direction * (source_target - waiting) <= -release_margin
                or source_idle
            ):
                self.inactive_collision_follow_source[state_index] = 0
                self.inactive_collision_follow_offset_rad[state_index] = 0.0
                self.inactive_collision_approach_direction[state_index] = 0.0

        if previous is not None:
            for left, right in zip(
                INACTIVE_COLLISION_CHAIN[:-1],
                INACTIVE_COLLISION_CHAIN[1:],
            ):
                propagated = False
                left_source = int(
                    self.inactive_collision_follow_source[left - 1]
                )
                right_source = int(
                    self.inactive_collision_follow_source[right - 1]
                )
                if (
                    left in active_set
                    and right not in active_set
                    and right_source == 0
                ):
                    follower, source = right, left
                elif (
                    right in active_set
                    and left not in active_set
                    and left_source == 0
                ):
                    follower, source = left, right
                elif left not in active_set and right not in active_set:
                    if left_source != 0 and right_source == 0:
                        if left_source == right:
                            continue
                        follower, source = right, left
                        propagated = True
                    elif right_source != 0 and left_source == 0:
                        if right_source == left:
                            continue
                        follower, source = left, right
                        propagated = True
                    else:
                        continue
                else:
                    continue

                gap = current[follower] - current[source]
                if propagated:
                    direction = float(
                        self.inactive_collision_approach_direction[source - 1]
                    )
                else:
                    previous_gap = previous[follower] - previous[source]
                    crossed = gap * previous_gap <= 0.0 and abs(
                        gap - previous_gap
                    ) > 1e-9
                    approaching = abs(gap) < abs(previous_gap) - 1e-9
                    if not (
                        abs(gap) <= tolerance
                        and (approaching or crossed)
                    ):
                        continue

                    source_motion = current[source] - previous[source]
                    if abs(source_motion) <= 1e-9:
                        source_motion = float(qdot[joint_indices[source]])
                    direction = float(np.sign(source_motion))
                if direction == 0.0:
                    continue
                self.inactive_collision_follow_source[follower - 1] = source
                self.inactive_collision_follow_offset_rad[follower - 1] = gap
                self.inactive_collision_approach_direction[follower - 1] = direction

        self.inactive_collision_avoidance_offsets_rad[:] = 0.0
        self.inactive_collision_avoidance_active = [False] * 5
        margin = max(
            0.0,
            float(self.cfg.inactive_collision_joint_limit_margin_rad),
        )
        command_targets = {}

        def command_target(finger):
            if finger in command_targets:
                return command_targets[finger]
            state_index = finger - 1
            source = int(self.inactive_collision_follow_source[state_index])
            if source == 0:
                command_targets[finger] = current[finger]
                return current[finger]
            lower, upper = self.finger_avoidance_joint_limits[finger]
            usable_margin = min(margin, 0.49 * (upper - lower))
            waiting = float(pre_grasp_pose[joint_indices[finger]])
            source_target = (
                command_target(source)
                + self.inactive_collision_follow_offset_rad[state_index]
            )
            direction = float(
                self.inactive_collision_approach_direction[state_index]
            )
            if direction * (source_target - waiting) < 0.0:
                source_target = waiting
            command_targets[finger] = float(
                np.clip(
                    source_target,
                    lower + usable_margin,
                    upper - usable_margin,
                )
            )
            return command_targets[finger]

        for finger in INACTIVE_COLLISION_CHAIN:
            state_index = finger - 1
            if self.inactive_collision_follow_source[state_index] == 0:
                continue
            target = command_target(finger)
            waiting = float(pre_grasp_pose[joint_indices[finger]])
            self.inactive_collision_avoidance_offsets_rad[state_index] = (
                target - waiting
            )
            self.inactive_collision_avoidance_active[state_index] = True

        self.inactive_collision_previous_joint_positions = current
        self.inactive_collision_min_clearance_m = -1.0
        return self.inactive_collision_avoidance_offsets_rad.copy()

    def _inactive_pre_grasp_pd(
        self,
        active_fingers,
        qdot,
        *,
        collision_avoidance_enabled=False,
    ):
        inactive_pd = np.zeros(JOINT_COUNT, dtype=np.float64)
        self.inactive_pd_target[:] = np.nan
        active_set = set(active_fingers)
        pre_grasp_pose = self._current_pre_grasp_pose()
        if collision_avoidance_enabled:
            avoidance_offsets = self._update_inactive_collision_avoidance(
                active_fingers,
                qdot,
            )
        else:
            self._reset_inactive_collision_avoidance()
            avoidance_offsets = self.inactive_collision_avoidance_offsets_rad

        for finger, idxs in FINGER_JOINT_INDEX.items():
            if finger in active_set:
                continue

            idxs = np.asarray(idxs, dtype=int)
            # Every unused finger waits at the complete selected pre-grasp
            # pose and can join the grasp without a preparation stage.
            target = pre_grasp_pose[idxs].copy()
            avoidance_joint = int(
                FINGER_AVOIDANCE_JOINT_LOCAL_INDEX[finger]
            )
            target[avoidance_joint] += avoidance_offsets[int(finger) - 1]

            kp = np.full(len(idxs), self.cfg.pose_kp, dtype=np.float64)
            kd = np.full(len(idxs), self.cfg.pose_kd, dtype=np.float64)
            limit = np.full(
                len(idxs),
                self.cfg.pose_pd_limit,
                dtype=np.float64,
            )
            avoidance_state_index = int(finger) - 1
            avoidance_moving = (
                collision_avoidance_enabled
                and (
                    self.inactive_collision_avoidance_active[
                        avoidance_state_index
                    ]
                    or abs(avoidance_offsets[avoidance_state_index]) > 1e-12
                )
            )
            if avoidance_moving:
                kp[avoidance_joint] = max(
                    0.0,
                    float(self.cfg.inactive_collision_pd_kp),
                )
                kd[avoidance_joint] = max(
                    0.0,
                    float(self.cfg.inactive_collision_pd_kd),
                )
                limit[avoidance_joint] = max(
                    0.0,
                    float(self.cfg.inactive_collision_pd_limit),
                )

            # Save the exact target that is actually sent to the inactive-finger
            # PD controller. This is used only by the periodic debug log.
            self.inactive_pd_target[idxs] = target

            pd, _ = pose_pd(
                target,
                self.hand_q[idxs],
                qdot[idxs],
                kp=kp,
                kd=kd,
                limit=limit,
            )
            if avoidance_moving:
                root_source = int(
                    self.inactive_collision_follow_source[
                        avoidance_state_index
                    ]
                )
                while self.inactive_collision_follow_source[root_source - 1]:
                    root_source = int(
                        self.inactive_collision_follow_source[root_source - 1]
                    )
                root_joint = int(
                    FINGER_JOINT_INDEX[root_source][
                        FINGER_AVOIDANCE_JOINT_LOCAL_INDEX[root_source]
                    ]
                )
                follower_velocity = qdot[idxs[avoidance_joint]]
                direction = float(
                    self.inactive_collision_approach_direction[
                        avoidance_state_index
                    ]
                )
                lower, upper = self.finger_avoidance_joint_limits[finger]
                margin = max(
                    0.0,
                    float(self.cfg.inactive_collision_joint_limit_margin_rad),
                )
                usable_margin = min(margin, 0.49 * (upper - lower))
                at_limit = (
                    target[avoidance_joint] <= lower + usable_margin + 1e-9
                    or target[avoidance_joint] >= upper - usable_margin - 1e-9
                )
                if direction * qdot[root_joint] > 0.0 and not at_limit:
                    relative_velocity = qdot[root_joint] - follower_velocity
                else:
                    relative_velocity = -follower_velocity
                pd[avoidance_joint] = np.clip(
                    kp[avoidance_joint]
                    * (
                        target[avoidance_joint]
                        - self.hand_q[idxs[avoidance_joint]]
                    )
                    + kd[avoidance_joint] * relative_velocity,
                    -limit[avoidance_joint],
                    limit[avoidance_joint],
                )
            inactive_pd[idxs] = pd

        return inactive_pd

    def _clear_transition_state(self):
        self.deferred_finger_count = None
        self.deferred_finger_count_at = None
        self._reset_inactive_collision_avoidance()

    def _reset_envelop_grasp(self):
        self.envelop_hold_pose = None
        self.envelop_started_at = None
        self.envelop_thumb_enabled = False
        self.envelop_thumb_start_at = None
        self.envelop_last_joint_stall_since = None
        self.envelop_last_info = {}

    def _start_envelop_grasp(self, now):
        self._clear_transition_state()
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
        local_joints = (
            ENVELOP_PINKY_TORQUE_LOCAL_JOINTS
            if finger == PINKY_FINGER_ID
            else ENVELOP_FINGER_TORQUE_LOCAL_JOINTS
        )
        joint_order_idx = local_joints.index(local_joint)
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
        #   t0 + 0*delay: index/middle/ring 2nd joints + pinky 3rd joint
        #   t0 + 1*delay: index/middle/ring J3 + pinky J4 + thumb J3
        #   t0 + 2*delay: index/middle/ring J4 + thumb J4
        # No qdot/stall condition is used for thumb triggering.
        for finger in ENVELOP_FINGER_ORDER:
            idxs = FINGER_JOINT_INDEX[finger]
            local_joints = (
                ENVELOP_PINKY_TORQUE_LOCAL_JOINTS
                if finger == PINKY_FINGER_ID
                else ENVELOP_FINGER_TORQUE_LOCAL_JOINTS
            )
            for local_joint in local_joints:
                joint_idx = idxs[local_joint]
                start_t = self._envelop_non_thumb_joint_start_time(finger, local_joint)
                if now >= start_t:
                    hold_mask[joint_idx] = False
                    tau[joint_idx] = float(self.cfg.envelop_non_thumb_tau_sign) * tau_level
                    active_non_thumb.append(joint_idx)

        thumb_idxs = FINGER_JOINT_INDEX[1]
        for order_idx, local_joint in enumerate(ENVELOP_THUMB_TORQUE_LOCAL_JOINTS):
            # Thumb J3 starts one stage before J4.
            joint_idx = thumb_idxs[local_joint]
            start_t = (
                self.envelop_started_at
                + (order_idx + 1) * self.cfg.envelop_joint_delay
            )
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
        if pose_type not in self.pose_type_targets:
            valid = ", ".join(map(str, self.pose_type_targets))
            raise ValueError(f"pose_type must be one of: {valid}")
        self.cancel_continuous_rotation()
        state, state_start, _ = self._apply_pose_type_command(pose_type, now)
        self.state = state
        self.state_start = state_start

    def _cancel_deferred_for_external_command(self) -> None:
        if self.deferred_finger_count is not None:
            self.deferred_finger_count = None
            self.deferred_finger_count_at = None

    def apply_grasp_type(self, requested_count: int, now: float, *, internal: bool = False) -> None:
        if requested_count < -1 or requested_count > 7:
            raise ValueError(
                "grasp_type must be one of -1, 0, 1, 2, 3, 4, 5, 6, 7"
            )

        if not internal:
            self.cancel_continuous_rotation()

        if requested_count == 7:
            self._start_card_grasp(now)
            return

        self.cancel_relative_rotation()
        self.cancel_relative_translation()
        self.cancel_card_grasp()
        self._reset_regular_force_balance_state()

        if not internal:
            self._cancel_deferred_for_external_command()

        command_count = requested_count

        if requested_count == -1:
            self.active_finger_count = 0
            self._clear_transition_state()
            self._reset_envelop_grasp()
            self.state = "NORMAL_POSE"
            self.state_start = now
            self._log("[COMMAND] grasp_type=-1 -> NORMAL_POSE")
            return

        if requested_count == 0:
            self.active_finger_count = 0
            self._clear_transition_state()
            self._reset_envelop_grasp()
            self.state = "PRE_GRASP_POSE"
            self.state_start = now
            self._log("[COMMAND] grasp_type=0 -> PRE_GRASP_POSE")
            return

        if requested_count == 6:
            self.active_finger_count = 6
            self._clear_transition_state()
            self._start_envelop_grasp(now)
            self.state = "ENVELOP_GRASP"
            self.state_start = now
            self._log(
                "[COMMAND] grasp_type=6 -> ENVELOP_GRASP "
                f"tau_per_joint={self.cfg.alpha1 * self.cfg.envelop_tau_scale:.4f}, "
                "joint_stage_order=(index/middle/ring J2+pinky J3) -> "
                "(index/middle/ring J3+pinky J4+thumb J3) -> "
                "(index/middle/ring J4+thumb J4)"
            )
            return

        self._reset_envelop_grasp()
        self._reset_inactive_collision_avoidance()

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
        self.use_fingers = target_fingers
        self.policy = GraspPolicy(self.use_fingers, self.cfg)
        self.active_finger_count = command_count
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
            self.deferred_finger_count is not None
            and self.deferred_finger_count_at is not None
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
        self._process_continuous_rotation(now)

        err = np.zeros(JOINT_COUNT, dtype=np.float64)
        grasp_tau = np.zeros(JOINT_COUNT, dtype=np.float64)
        translation_torques = np.zeros(JOINT_COUNT, dtype=np.float64)
        inactive_pd = np.zeros(JOINT_COUNT, dtype=np.float64)
        alpha = {}
        cg = np.zeros(3, dtype=np.float64)
        cv = np.zeros(3, dtype=np.float64)
        fingertip_positions = {}
        grasp_forces = {}
        translation_forces = {}
        rotation_forces = {}
        center_hold_forces = {}
        collision_forces = {}
        total_forces = {}
        effective_thumb_centroid_bias = 0.0
        relative_rotation_force_balance_blend = 0.0
        relative_rotation_control_mode = "idle"
        relative_translation_torque_target = 0.0
        relative_translation_force_scale = 0.0
        relative_translation_control_mode = "idle"
        relative_translation_dls_sigma_min = 0.0
        relative_translation_dls_condition = 0.0
        relative_translation_joint_error = np.zeros(
            JOINT_COUNT, dtype=np.float64
        )
        relative_translation_position_torques = np.zeros(
            JOINT_COUNT, dtype=np.float64
        )
        relative_translation_nullspace_grasp_torques = np.zeros(
            JOINT_COUNT, dtype=np.float64
        )
        self.inactive_pd_target[:] = np.nan

        if self.state == "NORMAL_POSE":
            tau, err = pose_pd(
                self.pose_type_targets[1],
                self.hand_q,
                qdot,
                kp=self.cfg.pose_kp,
                kd=self.cfg.pose_kd,
                limit=self.cfg.pose_pd_limit,
            )

        elif self.state == "PRE_GRASP_POSE":
            pre_rotation = (
                self.cfg.hand_side == "right" and self.pose_type in (5, 6)
            )
            blind_grasp_pre_rotation = pre_rotation and self.pose_type == 6
            pose_target = (
                self.continuous_rotation_pose_target
                if self.continuous_rotation_active
                else self._current_pre_grasp_pose()
            )
            tau, err = pose_pd(
                pose_target,
                self.hand_q,
                qdot,
                kp=(
                    self.cfg.blind_grasp_pre_rotation_pose_kp
                    if blind_grasp_pre_rotation
                    else self.cfg.pre_rotation_pose_kp
                    if pre_rotation
                    else self.cfg.pose_kp
                ),
                kd=(
                    self.cfg.blind_grasp_pre_rotation_pose_kd
                    if blind_grasp_pre_rotation
                    else self.cfg.pre_rotation_pose_kd
                    if pre_rotation
                    else self.cfg.pose_kd
                ),
                limit=(
                    self.cfg.pre_rotation_pose_pd_limit
                    if pre_rotation
                    else self.cfg.pose_pd_limit
                ),
            )
            if pre_rotation and not blind_grasp_pre_rotation:
                pinky_j1 = int(FINGER_JOINT_INDEX[5][0])
                pinky_tau, _ = pose_pd(
                    [pose_target[pinky_j1]],
                    [self.hand_q[pinky_j1]],
                    [qdot[pinky_j1]],
                    kp=self.cfg.pre_rotation_pinky_j1_kp,
                    kd=self.cfg.pre_rotation_pinky_j1_kd,
                    limit=self.cfg.pre_rotation_pinky_j1_tau_limit,
                )
                tau[pinky_j1] = pinky_tau[0]
        elif self.state == "CARD_GRASP":
            policy_result, inactive_pd, err = self._calc_card_grasp(qdot, now)
            grasp_tau = policy_result.tau.copy()
            alpha = dict(policy_result.alpha)
            cg = policy_result.cg.copy()
            cv = policy_result.cv.copy()
            fingertip_positions.update(
                _copy_finger_vectors(policy_result.fingertip_positions)
            )
            grasp_forces.update(
                _copy_finger_vectors(policy_result.grasp_forces)
            )
            rotation_forces.update(
                _copy_finger_vectors(policy_result.rotation_forces)
            )
            center_hold_forces.update(
                _copy_finger_vectors(policy_result.center_hold_forces)
            )
            collision_forces.update(
                _copy_finger_vectors(policy_result.collision_forces)
            )
            total_forces.update(
                _copy_finger_vectors(policy_result.total_forces)
            )
            tau = grasp_tau + inactive_pd

        elif self.state == "GROPED_GRASP":
            regular_grasp = 1 <= self.active_finger_count <= 5
            if regular_grasp:
                effective_thumb_centroid_bias = 0.0
                relative_rotation_force_balance_blend = 1.0
            else:
                effective_thumb_centroid_bias = float(
                    self.cfg.thumb_centroid_bias
                )

            using_force_balance_fallback = False
            if (
                regular_grasp
                and self.regular_force_balance_error_started_at is not None
            ):
                using_force_balance_fallback = True
                relative_rotation_force_balance_blend = 0.0
                policy_result = self._regular_force_balance_fail_closed_result(
                    now
                )
            else:
                try:
                    policy_result = self.policy.calc_grasp_tau(
                        self.hand_q,
                        alpha_distribution_mode=(
                            ALPHA_DISTRIBUTION_THUMB_DISTANCE_PROPORTIONAL
                        ),
                    )
                except ValueError as exc:
                    if not regular_grasp:
                        raise
                    self._log(
                        "[FORCE_DISTRIBUTION] proportional balance failed; "
                        "ramping the last valid command to zero and latching "
                        f"force_balance_error: {exc}"
                    )
                    self.regular_force_balance_error_started_at = float(now)
                    self.cancel_relative_translation()
                    self.cancel_relative_rotation()
                    self.relative_rotation_phase = "force_balance_error"
                    relative_rotation_force_balance_blend = 0.0
                    using_force_balance_fallback = True
                    policy_result = (
                        self._regular_force_balance_fail_closed_result(now)
                    )
            grasp_tau = policy_result.tau.copy()
            alpha = dict(policy_result.alpha)
            cg = policy_result.cg.copy()
            cv = policy_result.cv.copy()
            fingertip_positions.update(
                _copy_finger_vectors(policy_result.fingertip_positions)
            )
            grasp_forces.update(_copy_finger_vectors(policy_result.grasp_forces))
            rotation_forces.update(_copy_finger_vectors(policy_result.rotation_forces))
            center_hold_forces.update(_copy_finger_vectors(policy_result.center_hold_forces))
            collision_forces.update(_copy_finger_vectors(policy_result.collision_forces))
            total_forces.update(_copy_finger_vectors(policy_result.total_forces))

            if regular_grasp:
                base_total_forces = _copy_finger_vectors(total_forces)
                generic_rotation_forces = (
                    self._calc_relative_rotation_forces(
                        fingertip_positions,
                        now,
                        qdot,
                    )
                )
                rotation_forces.update(generic_rotation_forces)
                if self.relative_rotation_phase in {
                    "rotating",
                    "rotation_reached",
                }:
                    total_forces = _copy_finger_vectors(base_total_forces)
                    for finger in self.use_fingers:
                        total_forces[finger] += rotation_forces[finger]
                    grasp_tau = self.policy.calc_tau_from_total_forces(
                        self.hand_q,
                        total_forces,
                    )
                    if self.relative_rotation_phase == "rotating":
                        relative_rotation_control_mode = (
                            "cartesian_thumb_pivot_jacobian_transpose"
                        )
                    else:
                        relative_rotation_control_mode = (
                            "cartesian_thumb_pivot_hold"
                        )
                translation_forces.update(
                    self._calc_relative_translation_forces(
                        cg,
                        fingertip_positions,
                        now,
                        qdot,
                    )
                )
                if self.relative_translation_phase in {
                    "translating",
                    "translation_reached",
                }:
                    grasp_tau_without_translation = grasp_tau.copy()
                    for finger in self.use_fingers:
                        total_forces[finger] += translation_forces[finger]
                    grasp_tau = self.policy.calc_tau_from_total_forces(
                        self.hand_q,
                        total_forces,
                    )
                    translation_torques = (
                        grasp_tau - grasp_tau_without_translation
                    )
                    relative_translation_position_torques = (
                        translation_torques.copy()
                    )
                    if self.relative_translation_phase in {
                        "translating",
                        "translation_reached",
                    }:
                        relative_translation_control_mode = (
                            "cartesian_fingertip_jacobian_transpose"
                        )

            if (
                regular_grasp
                and not using_force_balance_fallback
            ):
                # Keep the fail-closed cache free of translation and rotation
                # manipulation. Only the stable regular grasp command may be
                # replayed and faded after a later balance failure.
                cached_total_forces = _copy_finger_vectors(base_total_forces)
                self.last_regular_policy_result = replace(
                    policy_result,
                    tau=self.policy.calc_tau_from_total_forces(
                        self.hand_q,
                        cached_total_forces,
                    ),
                    alpha={
                        int(finger): float(np.linalg.norm(force))
                        for finger, force in grasp_forces.items()
                    },
                    grasp_forces=_copy_finger_vectors(
                        policy_result.grasp_forces
                    ),
                    rotation_forces=_copy_finger_vectors(
                        policy_result.rotation_forces
                    ),
                    center_hold_forces=_copy_finger_vectors(
                        policy_result.center_hold_forces
                    ),
                    collision_forces=_copy_finger_vectors(
                        policy_result.collision_forces
                    ),
                    total_forces=cached_total_forces,
                )
                self.last_regular_policy_fingers = tuple(self.use_fingers)

            inactive_control_fingers = list(self.use_fingers)
            inactive_pd = self._inactive_pre_grasp_pd(
                inactive_control_fingers,
                qdot,
                collision_avoidance_enabled=regular_grasp,
            )
            tau = grasp_tau + inactive_pd

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
            translation_torques=translation_torques.copy(),
            inactive_pd=inactive_pd.copy(),
            alpha={int(k): float(v) for k, v in alpha.items()},
            cg=cg.copy(),
            cv=cv.copy(),
            fingertip_positions=_copy_finger_vectors(fingertip_positions),
            grasp_forces=_copy_finger_vectors(grasp_forces),
            translation_forces=_copy_finger_vectors(translation_forces),
            rotation_forces=_copy_finger_vectors(rotation_forces),
            center_hold_forces=_copy_finger_vectors(center_hold_forces),
            collision_forces=_copy_finger_vectors(collision_forces),
            total_forces=_copy_finger_vectors(total_forces),
            use_fingers=list(self.use_fingers),
            active_finger_count=int(self.active_finger_count),
            inactive_pd_target=self.inactive_pd_target.copy(),
            inactive_collision_min_clearance_m=float(
                self.inactive_collision_min_clearance_m
            ),
            inactive_collision_avoidance_offsets_rad=(
                self.inactive_collision_avoidance_offsets_rad.copy()
            ),
            inactive_collision_avoidance_active=list(
                self.inactive_collision_avoidance_active
            ),
            envelop_info=dict(self.envelop_last_info),
            relative_rotation_phase=str(self.relative_rotation_phase),
            relative_rotation_target_rad=float(self.relative_rotation_target_rad),
            relative_rotation_current_rad=float(
                self.relative_rotation_current_rad
            ),
            relative_rotation_error_rad=float(
                self.relative_rotation_error_rad
            ),
            relative_rotation_angular_velocity=float(
                self.relative_rotation_angular_velocity
            ),
            relative_rotation_command_moment=float(
                self.relative_rotation_command_moment
            ),
            relative_rotation_axis=self.relative_rotation_axis.copy(),
            relative_rotation_pivot=self.relative_rotation_pivot.copy(),
            relative_rotation_control_mode=str(
                relative_rotation_control_mode
            ),
            relative_translation_phase=str(self.relative_translation_phase),
            relative_translation_start_centroid=(
                self.relative_translation_start_centroid.copy()
            ),
            relative_translation_target_centroid=(
                self.relative_translation_target_centroid.copy()
            ),
            relative_translation_delta=self.relative_translation_delta.copy(),
            relative_translation_error=self.relative_translation_error.copy(),
            relative_translation_centroid_velocity=(
                self.relative_translation_centroid_velocity.copy()
            ),
            relative_translation_command_force=(
                self.relative_translation_command_force.copy()
            ),
            relative_translation_torque_target=float(
                relative_translation_torque_target
            ),
            relative_translation_force_scale=float(
                relative_translation_force_scale
            ),
            relative_translation_control_mode=str(
                relative_translation_control_mode
            ),
            relative_translation_dls_sigma_min=float(
                relative_translation_dls_sigma_min
            ),
            relative_translation_dls_condition=float(
                relative_translation_dls_condition
            ),
            relative_translation_joint_error=(
                relative_translation_joint_error.copy()
            ),
            relative_translation_position_torques=(
                relative_translation_position_torques.copy()
            ),
            relative_translation_nullspace_grasp_torques=(
                relative_translation_nullspace_grasp_torques.copy()
            ),
            effective_thumb_centroid_bias=float(effective_thumb_centroid_bias),
            relative_rotation_force_balance_blend=float(
                relative_rotation_force_balance_blend
            ),
        )
