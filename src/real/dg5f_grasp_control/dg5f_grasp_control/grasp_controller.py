from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional

import numpy as np

from dg5f_grasp_control.config import RuntimeConfig
from dg5f_grasp_control.control_utils import pose_pd
from dg5f_grasp_control.grasp_policy import (
    ALPHA_DISTRIBUTION_LEGACY,
    ALPHA_DISTRIBUTION_THUMB_DISTANCE_PROPORTIONAL,
    BASE_JOINT_TAU_LIMIT,
    GraspPolicy,
    GraspPolicyResult,
)
from dg5f_grasp_control.hand_model import (
    FINGER_AVOIDANCE_JOINT_LIMITS,
    FINGER_AVOIDANCE_JOINT_LOCAL_INDEX,
    FINGER_JOINT_INDEX,
    GRASP_TAU_SIGN,
    JOINT_COUNT,
    selected_fingers,
)
from dg5f_grasp_control.kinematics import (
    capsule_segments_clearance,
    finger_capsule_segments,
    tip_jacobian,
    tip_position,
)
from dg5f_grasp_control.poses import (
    HAND_NORMAL_POSE,
    HAND_PRE_GRASP_POSE,
    POSE_TYPE_TARGETS,
)

FINGER_SWITCH_VIA_THREE_DELAY = 0.5

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
    rotation_enabled: bool = False
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
    g7_phase: str = "idle"
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

        self.inactive_pd_target = np.full(JOINT_COUNT, np.nan, dtype=np.float64)
        self.inactive_collision_avoidance_offsets_rad = np.zeros(
            5,
            dtype=np.float64,
        )
        self.inactive_collision_avoidance_active = [False] * 5
        self.inactive_collision_avoidance_direction = np.zeros(
            5,
            dtype=np.float64,
        )
        self.inactive_collision_min_clearance_m = -1.0

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
        self.relative_rotation_last_time = None
        self.relative_rotation_reference_progress = 0.0
        # Relative task-space motion state. The grasp policy keeps the normal
        # holding force active in the null space while a centroid DLS
        # joint-position controller tracks the stored Cartesian target.
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
        self.relative_translation_last_fingertips = {}
        self.relative_translation_fingertip_velocities = {}
        self.relative_translation_contact_weights = {}
        self.relative_translation_max_axis_fingertip_error = 0.0
        self.relative_translation_reference_progress = 0.0
        self.relative_translation_control_axis_error = 0.0
        self.relative_translation_control_axis_drive_force = 0.0
        self.relative_translation_started_at = None
        self.relative_translation_last_time = None
        self.relative_translation_last_centroid = None
        self.relative_translation_reached_since = None
        self.regular_force_balance_error_started_at = None
        self.last_regular_policy_result = None
        self.last_regular_policy_fingers = ()

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
        self.relative_rotation_last_time = None
        self.relative_rotation_reference_progress = 0.0

    def cancel_relative_translation(self) -> None:
        """Cancel the stored Cartesian translation target and DLS control."""

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
        self.relative_translation_last_fingertips = {}
        self.relative_translation_fingertip_velocities = {}
        self.relative_translation_contact_weights = {}
        self.relative_translation_max_axis_fingertip_error = 0.0
        self.relative_translation_reference_progress = 0.0
        self.relative_translation_control_axis_error = 0.0
        self.relative_translation_control_axis_drive_force = 0.0
        self.relative_translation_started_at = None
        self.relative_translation_last_time = None
        self.relative_translation_last_centroid = None
        self.relative_translation_reached_since = None

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

    def prepare_relative_rotation(self, angle_rad: float, now: float) -> bool:
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
        self.relative_rotation_last_time = now
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
        try:
            contact_weights = self._calc_centroid_contact_weights(
                start,
                start_fingertips,
            )
        except ValueError as exc:
            self._log(
                "[RELATIVE_TRANSLATION] ignored: cannot construct Cartesian "
                f"contact weights: {exc}"
            )
            return False

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
        self.relative_translation_last_fingertips = {
            finger: position.copy()
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
        self.relative_translation_last_time = now
        self.relative_translation_last_centroid = start.copy()
        self.relative_translation_reached_since = None
        self.relative_translation_phase = "translating"
        self._log(
            "[RELATIVE_TRANSLATION] centroid DLS/null-space grasp target started "
            f"delta_link_base_mm={np.round(1000.0 * delta_hand, 3).tolist()}"
        )
        return True

    @staticmethod
    def _calc_centroid_contact_weights(
        cg: np.ndarray,
        tip_positions: Dict[int, np.ndarray],
    ) -> Dict[int, float]:
        """Find non-negative weights whose weighted contact point is ``cg``.

        Using these weights for identical fingertip position errors produces
        the requested resultant at the geometric centroid with zero initial
        moment. Once individual errors differ, the independent fingertip PD
        terms are allowed to create the restoring moment needed to preserve
        the captured grasp shape.
        """

        fingers = list(tip_positions)
        if not fingers:
            raise ValueError("no active contacts")
        points = np.column_stack(
            [np.asarray(tip_positions[finger], dtype=np.float64) for finger in fingers]
        )
        cg = np.asarray(cg, dtype=np.float64)
        if (
            points.shape != (3, len(fingers))
            or cg.shape != (3,)
            or not np.all(np.isfinite(points))
            or not np.all(np.isfinite(cg))
        ):
            raise ValueError("contacts and centroid must be finite 3-vectors")

        constraints = np.vstack((points, np.ones((1, len(fingers)))))
        target = np.concatenate((cg, np.ones(1, dtype=np.float64)))
        reference = np.full(len(fingers), 1.0 / len(fingers), dtype=np.float64)
        tolerance = 1e-9 * max(1.0, float(np.linalg.norm(target)))
        best = None

        # At most five contacts are active, so all non-empty subsets are cheap
        # to enumerate and avoid introducing a constrained-solver dependency.
        for mask in range(1, 1 << len(fingers)):
            active = [
                index
                for index in range(len(fingers))
                if mask & (1 << index)
            ]
            active_constraints = constraints[:, active]
            active_reference = reference[active]
            try:
                correction = np.linalg.lstsq(
                    active_constraints,
                    target - active_constraints @ active_reference,
                    rcond=None,
                )[0]
            except np.linalg.LinAlgError:
                continue
            active_weights = active_reference + correction
            if (
                not np.all(np.isfinite(active_weights))
                or np.any(active_weights < -tolerance)
                or np.linalg.norm(active_constraints @ active_weights - target)
                > tolerance
            ):
                continue
            candidate = np.zeros(len(fingers), dtype=np.float64)
            candidate[active] = np.maximum(active_weights, 0.0)
            residual = float(np.linalg.norm(constraints @ candidate - target))
            if residual > tolerance:
                continue
            score = float(np.linalg.norm(candidate - reference))
            if best is None or score < best[0]:
                best = (score, candidate)

        if best is None:
            raise ValueError("geometric centroid is outside the contact hull")
        return {
            int(finger): float(weight)
            for finger, weight in zip(fingers, best[1])
        }

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

        dt = None
        if self.relative_rotation_last_time is not None:
            dt = now - float(self.relative_rotation_last_time)
        if dt is not None and dt > 1e-6:
            previous_wrapped = (
                wrapped_angle
                if self.relative_rotation_last_wrapped_angle is None
                else float(self.relative_rotation_last_wrapped_angle)
            )
            angle_step = self._wrap_angle(wrapped_angle - previous_wrapped)
            self.relative_rotation_current_rad += angle_step
            raw_angular_velocity = angle_step / dt
            velocity_alpha = float(
                np.clip(self.cfg.relative_rotation_velocity_alpha, 0.0, 1.0)
            )
            self.relative_rotation_angular_velocity = (
                (1.0 - velocity_alpha)
                * self.relative_rotation_angular_velocity
                + velocity_alpha * raw_angular_velocity
            )
        else:
            self.relative_rotation_current_rad = wrapped_angle

        self.relative_rotation_last_wrapped_angle = wrapped_angle
        self.relative_rotation_last_time = now
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

        position_kp = float(self.cfg.relative_rotation_position_kp)
        position_kd = float(self.cfg.relative_rotation_position_kd)
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
        thumb_indices = np.asarray(FINGER_JOINT_INDEX[1], dtype=int)
        thumb_velocity = (
            tip_jacobian(
                self.hand_q,
                1,
                eps=self.cfg.jacobian_eps,
            )
            @ qdot[thumb_indices]
        )
        driven_fingers = [
            finger
            for finger in self.relative_rotation_start_fingertips
            if finger != 1
        ]
        if not driven_fingers:
            self.relative_rotation_phase = "rotation_error"
            self._log("[RELATIVE_ROTATION] stopped: no non-thumb driven finger")
            return rotation_forces
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

            indices = np.asarray(FINGER_JOINT_INDEX[finger], dtype=int)
            fingertip_velocity = (
                tip_jacobian(
                    self.hand_q,
                    finger,
                    eps=self.cfg.jacobian_eps,
                )
                @ qdot[indices]
            )
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
    ) -> Dict[int, np.ndarray]:
        """Update target state, virtual-force diagnostics, and shape forces."""

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
        now = float(now)
        expected_fingers = set(self.relative_translation_target_fingertips)
        if (
            cg.shape != (3,)
            or not np.all(np.isfinite(cg))
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

        dt = None
        if (
            self.relative_translation_last_centroid is not None
            and self.relative_translation_last_time is not None
        ):
            dt = now - float(self.relative_translation_last_time)
            if dt > 1e-6:
                raw_centroid_velocity = (
                    cg - self.relative_translation_last_centroid
                ) / dt
                velocity_alpha = float(
                    np.clip(self.cfg.relative_translation_velocity_alpha, 0.0, 1.0)
                )
                self.relative_translation_centroid_velocity = (
                    (1.0 - velocity_alpha)
                    * self.relative_translation_centroid_velocity
                    + velocity_alpha * raw_centroid_velocity
                )
                for finger, position in tip_positions.items():
                    previous = self.relative_translation_last_fingertips.get(finger)
                    if previous is None:
                        continue
                    raw_fingertip_velocity = (position - previous) / dt
                    previous_velocity = (
                        self.relative_translation_fingertip_velocities.get(
                            finger,
                            np.zeros(3, dtype=np.float64),
                        )
                    )
                    self.relative_translation_fingertip_velocities[finger] = (
                        (1.0 - velocity_alpha) * previous_velocity
                        + velocity_alpha * raw_fingertip_velocity
                    )
        self.relative_translation_last_centroid = cg.copy()
        self.relative_translation_last_fingertips = {
            finger: position.copy()
            for finger, position in tip_positions.items()
        }
        self.relative_translation_last_time = now
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
            abs(float(np.dot(translation_axis, error)))
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
                f"max_axis_tip_error_mm="
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
        maximum_axis_fingertip_velocity = max(
            (
                abs(float(np.dot(translation_axis, velocity)))
                for velocity in self.relative_translation_fingertip_velocities.values()
            ),
            default=0.0,
        )
        axis_velocity = max(
            abs(
                float(
                    np.dot(
                        translation_axis,
                        self.relative_translation_centroid_velocity,
                    )
                )
            ),
            maximum_axis_fingertip_velocity,
        )
        inside_tolerance = (
            reference_progress >= 1.0 - 1e-12
            and axis_position_error <= position_tolerance
            and self.relative_translation_max_axis_fingertip_error
            <= position_tolerance
            and axis_velocity <= velocity_tolerance
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
                    f"max_axis_tip_error_mm="
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
        shape_kp = float(self.cfg.relative_translation_shape_kp)
        shape_kd = float(self.cfg.relative_translation_shape_kd)
        gains = (kp, kd, shape_kp, shape_kd)
        if not all(np.isfinite(gain) and gain >= 0.0 for gain in gains):
            self.relative_translation_phase = "translation_error"
            self._log("[RELATIVE_TRANSLATION] stopped: invalid Cartesian gains")
            return zero

        centroid_velocity = self.relative_translation_centroid_velocity
        axis_error = (
            translation_axis
            * float(np.dot(translation_axis, control_centroid_error))
        )
        axis_velocity = (
            translation_axis
            * float(np.dot(translation_axis, centroid_velocity))
        )
        object_force = kp * axis_error - kd * axis_velocity
        self.relative_translation_control_axis_drive_force = float(
            np.dot(translation_axis, object_force)
        )

        raw_shape_forces = {}
        for finger, error in control_fingertip_errors.items():
            weight = float(self.relative_translation_contact_weights[finger])
            velocity = self.relative_translation_fingertip_velocities.get(
                finger,
                np.zeros(3, dtype=np.float64),
            )
            shape_error = error - control_centroid_error
            shape_velocity = velocity - centroid_velocity
            raw_shape_forces[finger] = weight * (
                shape_kp * shape_error - shape_kd * shape_velocity
            )
        raw_shape_resultant = sum(
            raw_shape_forces.values(),
            np.zeros(3, dtype=np.float64),
        )

        forces = {}
        for finger, raw_shape_force in raw_shape_forces.items():
            weight = float(self.relative_translation_contact_weights[finger])
            # The shape term is projected into the internal-force subspace so
            # it can restore relative fingertip geometry without shaking the
            # object through an unintended resultant.
            shape_force = raw_shape_force - weight * raw_shape_resultant
            force = weight * object_force + shape_force
            if force.shape != (3,) or not np.all(np.isfinite(force)):
                self.relative_translation_phase = "translation_error"
                self._log(
                    "[RELATIVE_TRANSLATION] stopped: non-finite hybrid task force"
                )
                return zero
            forces[finger] = force

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
                shape_force = (
                    forces[finger]
                    - float(self.relative_translation_contact_weights[finger])
                    * (scale * object_force)
                )
                self.relative_translation_shape_forces[finger] = shape_force
        else:
            for finger, force in forces.items():
                self.relative_translation_shape_forces[finger] = (
                    force
                    - float(self.relative_translation_contact_weights[finger])
                    * object_force
                )
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

    def _calc_relative_translation_joint_control(
        self,
        *,
        base_total_forces: Dict[int, np.ndarray],
        base_grasp_tau: np.ndarray,
        qdot: np.ndarray,
    ):
        """Combine command-axis centroid DLS and null-space grasp torque.

        Only centroid motion along the requested direction has task priority.
        Orthogonal centroid directions remain unconstrained. Existing grasp
        and zero-resultant shape torque are projected out of the 1-D task.
        """

        zero = np.zeros(JOINT_COUNT, dtype=np.float64)
        active_phases = {"translating", "translation_reached"}
        if self.relative_translation_phase not in active_phases:
            return (
                base_grasp_tau.copy(),
                zero.copy(),
                zero.copy(),
                zero.copy(),
                0.0,
                0.0,
            )

        fingers = list(self.use_fingers)
        if (
            not fingers
            or set(self.relative_translation_contact_weights) != set(fingers)
        ):
            self.relative_translation_phase = "translation_error"
            self._log("[RELATIVE_TRANSLATION] stopped: invalid centroid weights")
            return (
                base_grasp_tau.copy(),
                zero.copy(),
                zero.copy(),
                zero.copy(),
                0.0,
                0.0,
            )

        active_indices = np.concatenate(
            [np.asarray(FINGER_JOINT_INDEX[finger], dtype=int) for finger in fingers]
        )
        centroid_jacobian = np.hstack(
            [
                float(self.relative_translation_contact_weights[finger])
                * tip_jacobian(
                    self.hand_q,
                    finger,
                    eps=self.cfg.jacobian_eps,
                )
                for finger in fingers
            ]
        )
        translation_distance = float(
            np.linalg.norm(self.relative_translation_delta)
        )
        if not np.isfinite(translation_distance) or translation_distance <= 1e-12:
            self.relative_translation_phase = "translation_error"
            self._log("[RELATIVE_TRANSLATION] stopped: invalid translation axis")
            return (
                base_grasp_tau.copy(),
                zero.copy(),
                zero.copy(),
                zero.copy(),
                0.0,
                float("inf"),
            )
        translation_axis = self.relative_translation_delta / translation_distance
        axis_jacobian = translation_axis.reshape(1, 3) @ centroid_jacobian
        singular_values = np.linalg.svd(axis_jacobian, compute_uv=False)
        sigma_min = float(singular_values[-1]) if singular_values.size else 0.0
        condition = 1.0 if sigma_min > 1e-12 else float("inf")

        damping = float(self.cfg.relative_translation_dls_damping)
        rcond = float(self.cfg.relative_translation_nullspace_rcond)
        joint_kp = float(self.cfg.relative_translation_joint_kp)
        joint_kd = float(self.cfg.relative_translation_joint_kd)
        correction_limit = float(
            self.cfg.relative_translation_joint_correction_limit_rad
        )
        position_torque_limit = float(
            self.cfg.relative_translation_position_torque_limit
        )
        nullspace_grasp_gain = float(
            self.cfg.relative_translation_nullspace_grasp_gain
        )
        values = (
            damping,
            rcond,
            joint_kp,
            joint_kd,
            correction_limit,
            position_torque_limit,
            nullspace_grasp_gain,
        )
        if (
            not all(np.isfinite(value) for value in values)
            or damping < 0.0
            or rcond < 0.0
            or joint_kp < 0.0
            or joint_kd < 0.0
            or correction_limit <= 0.0
            or position_torque_limit <= 0.0
            or nullspace_grasp_gain < 0.0
        ):
            self.relative_translation_phase = "translation_error"
            self._log("[RELATIVE_TRANSLATION] stopped: invalid DLS/null-space gains")
            return (
                base_grasp_tau.copy(),
                zero.copy(),
                zero.copy(),
                zero.copy(),
                sigma_min,
                condition,
            )

        reference_centroid = (
            self.relative_translation_start_centroid
            + self.relative_translation_reference_progress
            * self.relative_translation_delta
        )
        current_centroid = (
            self.relative_translation_target_centroid
            - self.relative_translation_error
        )
        control_error = float(
            np.dot(
                translation_axis,
                reference_centroid - current_centroid,
            )
        )

        try:
            denominator = float(
                (axis_jacobian @ axis_jacobian.T).item()
                + damping * damping
            )
            if not np.isfinite(denominator) or denominator <= 1e-15:
                raise np.linalg.LinAlgError("degenerate command-axis Jacobian")
            dls_inverse = axis_jacobian.T / denominator
            nullspace_inverse = np.linalg.pinv(
                axis_jacobian,
                rcond=rcond,
            )
        except np.linalg.LinAlgError:
            self.relative_translation_phase = "translation_error"
            self._log("[RELATIVE_TRANSLATION] stopped: centroid Jacobian solve failed")
            return (
                base_grasp_tau.copy(),
                zero.copy(),
                zero.copy(),
                zero.copy(),
                sigma_min,
                condition,
            )

        joint_error_active = dls_inverse[:, 0] * control_error
        maximum_joint_error = float(np.max(np.abs(joint_error_active)))
        if maximum_joint_error > correction_limit:
            joint_error_active *= correction_limit / maximum_joint_error

        qdot_active = np.asarray(qdot, dtype=np.float64)[active_indices]
        axis_task_velocity = float((axis_jacobian @ qdot_active).item())
        task_joint_velocity = dls_inverse[:, 0] * axis_task_velocity
        position_tau_active = (
            joint_kp * joint_error_active - joint_kd * task_joint_velocity
        )
        position_tau_active = np.clip(
            position_tau_active,
            -position_torque_limit,
            position_torque_limit,
        )

        internal_forces = _copy_finger_vectors(base_total_forces)
        for finger, shape_force in self.relative_translation_shape_forces.items():
            internal_forces[finger] += np.asarray(shape_force, dtype=np.float64)
        internal_tau = self.policy.calc_tau_from_total_forces(
            self.hand_q,
            internal_forces,
        )
        null_projector = (
            np.eye(len(active_indices), dtype=np.float64)
            - nullspace_inverse @ axis_jacobian
        )
        projected_grasp_tau_active = (
            nullspace_grasp_gain
            * null_projector.T
            @ internal_tau[active_indices]
        )
        # Avoid a torque step when a move starts. The same smooth reference
        # progress transitions the secondary grasp/shape command into the
        # strict centroid-task null space.
        hierarchy_blend = float(
            np.clip(self.relative_translation_reference_progress, 0.0, 1.0)
        )
        nullspace_tau_active = (
            (1.0 - hierarchy_blend) * base_grasp_tau[active_indices]
            + hierarchy_blend * projected_grasp_tau_active
        )

        combined_tau = base_grasp_tau.copy()
        combined_tau[active_indices] = (
            position_tau_active + nullspace_tau_active
        )
        combined_tau = self._clip_regular_grasp_tau(combined_tau)

        joint_error = zero.copy()
        position_tau = zero.copy()
        nullspace_tau = zero.copy()
        joint_error[active_indices] = joint_error_active
        position_tau[active_indices] = position_tau_active
        nullspace_tau[active_indices] = nullspace_tau_active
        return (
            combined_tau,
            joint_error,
            position_tau,
            nullspace_tau,
            sigma_min,
            condition,
        )

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
        self.cancel_relative_rotation()
        self.cancel_relative_translation()
        self._reset_regular_force_balance_state()
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

    def _reset_inactive_collision_avoidance(self):
        self.inactive_collision_avoidance_offsets_rad[:] = 0.0
        self.inactive_collision_avoidance_active = [False] * 5
        self.inactive_collision_avoidance_direction[:] = 0.0
        self.inactive_collision_min_clearance_m = -1.0

    @staticmethod
    def _inactive_clearance_to_avoidance_sources(
        inactive_segments,
        inactive_finger,
        source_fingers,
        segment_cache,
        capsule_radius,
    ):
        neighbors = [
            int(finger)
            for finger in source_fingers
            if abs(int(finger) - int(inactive_finger)) == 1
        ]
        if not neighbors:
            return float("inf")

        return min(
            capsule_segments_clearance(
                inactive_segments,
                segment_cache[active_finger],
                capsule_radius,
            )
            for active_finger in neighbors
        )

    def _update_inactive_collision_avoidance(self, active_fingers, qdot):
        """Update rate-limited joint-1 offsets with chained avoidance."""

        active_set = {int(finger) for finger in active_fingers}
        # An inactive finger that is moving away from an active finger becomes
        # an avoidance source for the next inactive neighbor. This propagates
        # middle -> ring -> pinky clearance without making fingers spread in
        # their undisturbed pre-grasp pose.
        avoidance_sources = set(active_set)
        if not bool(self.cfg.inactive_collision_avoidance_enable):
            self._reset_inactive_collision_avoidance()
            return self.inactive_collision_avoidance_offsets_rad.copy()

        prediction_sec = max(
            0.0,
            float(self.cfg.inactive_collision_prediction_sec),
        )
        predicted_q = self.hand_q + prediction_sec * np.asarray(
            qdot,
            dtype=np.float64,
        )
        radius = max(
            0.0,
            float(self.cfg.inactive_collision_capsule_radius_m),
        )
        first_segment = int(
            np.clip(self.cfg.inactive_collision_first_segment, 0, 3)
        )
        current_segment_cache = {
            finger: finger_capsule_segments(self.hand_q, finger)[
                first_segment:
            ]
            for finger in FINGER_JOINT_INDEX
        }
        predicted_segment_cache = {
            finger: finger_capsule_segments(predicted_q, finger)[
                first_segment:
            ]
            for finger in FINGER_JOINT_INDEX
        }
        activation = max(
            0.0,
            float(self.cfg.inactive_collision_activation_clearance_m),
        )
        critical = min(
            float(self.cfg.inactive_collision_critical_clearance_m),
            activation - 1e-6,
        )
        release = activation + max(
            0.0,
            float(self.cfg.inactive_collision_release_hysteresis_m),
        )
        maximum_offset = max(
            0.0,
            float(self.cfg.inactive_collision_max_joint1_offset_rad),
        )
        gradient_eps = max(
            1e-6,
            float(self.cfg.inactive_collision_gradient_eps_rad),
        )
        minimum_delta = max(
            0.0,
            float(self.cfg.inactive_collision_direction_min_delta_m),
        )
        rate = max(
            0.0,
            float(self.cfg.inactive_collision_joint1_target_rate_radps),
        )
        maximum_step = rate * max(float(self.cfg.dt), 1e-6)
        margin = max(
            0.0,
            float(self.cfg.inactive_collision_joint_limit_margin_rad),
        )
        pre_grasp_pose = self._current_pre_grasp_pose()
        observed_clearances = []

        for finger in FINGER_JOINT_INDEX:
            state_index = int(finger) - 1
            if finger in active_set or finger == 1:
                self.inactive_collision_avoidance_offsets_rad[state_index] = 0.0
                self.inactive_collision_avoidance_active[state_index] = False
                self.inactive_collision_avoidance_direction[state_index] = 0.0
                continue

            current_clearance = self._inactive_clearance_to_avoidance_sources(
                current_segment_cache[finger],
                finger,
                avoidance_sources,
                current_segment_cache,
                radius,
            )
            predicted_clearance = (
                self._inactive_clearance_to_avoidance_sources(
                    predicted_segment_cache[finger],
                    finger,
                    avoidance_sources,
                    predicted_segment_cache,
                    radius,
            )
            )
            clearance = min(current_clearance, predicted_clearance)
            if np.isfinite(clearance):
                observed_clearances.append(float(clearance))
            else:
                self.inactive_collision_avoidance_active[state_index] = False

            if (
                not self.inactive_collision_avoidance_active[state_index]
                and clearance < activation
            ):
                self.inactive_collision_avoidance_active[state_index] = True
            elif (
                self.inactive_collision_avoidance_active[state_index]
                and clearance >= release
            ):
                self.inactive_collision_avoidance_active[state_index] = False

            desired_offset = 0.0
            if self.inactive_collision_avoidance_active[state_index]:
                proximity = float(
                    np.clip(
                        (release - clearance) / max(release - critical, 1e-6),
                        0.0,
                        1.0,
                    )
                )
                proximity = proximity * proximity * (3.0 - 2.0 * proximity)
                direction_q = (
                    self.hand_q
                    if current_clearance <= predicted_clearance
                    else predicted_q
                ).copy()
                direction_segment_cache = (
                    current_segment_cache
                    if current_clearance <= predicted_clearance
                    else predicted_segment_cache
                )
                local_joint_index = int(
                    FINGER_AVOIDANCE_JOINT_LOCAL_INDEX[finger]
                )
                joint_index = int(
                    FINGER_JOINT_INDEX[finger][local_joint_index]
                )
                lower, upper = FINGER_AVOIDANCE_JOINT_LIMITS[finger]
                usable_margin = min(margin, 0.49 * (upper - lower))
                lower += usable_margin
                upper -= usable_margin

                plus_q = direction_q.copy()
                minus_q = direction_q.copy()
                plus_q[joint_index] = min(
                    upper,
                    plus_q[joint_index] + gradient_eps,
                )
                minus_q[joint_index] = max(
                    lower,
                    minus_q[joint_index] - gradient_eps,
                )
                plus_clearance = (
                    self._inactive_clearance_to_avoidance_sources(
                        finger_capsule_segments(plus_q, finger)[first_segment:],
                        finger,
                        avoidance_sources,
                        direction_segment_cache,
                        radius,
                )
                )
                minus_clearance = (
                    self._inactive_clearance_to_avoidance_sources(
                        finger_capsule_segments(minus_q, finger)[first_segment:],
                        finger,
                        avoidance_sources,
                        direction_segment_cache,
                    radius,
                )
                )
                clearance_delta = plus_clearance - minus_clearance
                if abs(clearance_delta) >= minimum_delta:
                    self.inactive_collision_avoidance_direction[state_index] = (
                        1.0 if clearance_delta > 0.0 else -1.0
                    )

                desired_offset = (
                    self.inactive_collision_avoidance_direction[state_index]
                    * maximum_offset
                    * proximity
                )
                pre_grasp_joint = float(pre_grasp_pose[joint_index])
                desired_target = float(
                    np.clip(
                        pre_grasp_joint + desired_offset,
                        lower,
                        upper,
                    )
                )
                desired_offset = desired_target - pre_grasp_joint

            previous_offset = float(
                self.inactive_collision_avoidance_offsets_rad[state_index]
            )
            offset_step = float(
                np.clip(
                    desired_offset - previous_offset,
                    -maximum_step,
                    maximum_step,
                )
            )
            self.inactive_collision_avoidance_offsets_rad[state_index] = (
                previous_offset + offset_step
            )
            offset = float(
                self.inactive_collision_avoidance_offsets_rad[state_index]
            )
            if (
                not self.inactive_collision_avoidance_active[state_index]
                and abs(offset) <= 1e-12
            ):
                self.inactive_collision_avoidance_direction[state_index] = 0.0

            if (
                self.inactive_collision_avoidance_active[state_index]
                or abs(offset) > 1e-12
            ):
                # Let the following inactive finger see both the measured
                # velocity prediction and where this avoidance command can
                # move the source finger during the same preview horizon.
                # Keeping both capsule sets makes the check conservative.
                local_joint_index = int(
                    FINGER_AVOIDANCE_JOINT_LOCAL_INDEX[finger]
                )
                joint_index = int(
                    FINGER_JOINT_INDEX[finger][local_joint_index]
                )
                lower, upper = FINGER_AVOIDANCE_JOINT_LIMITS[finger]
                usable_margin = min(margin, 0.49 * (upper - lower))
                lower += usable_margin
                upper -= usable_margin
                preview_target = float(
                    np.clip(
                        pre_grasp_pose[joint_index] + desired_offset,
                        lower,
                        upper,
                    )
                )
                preview_q = predicted_q.copy()
                preview_step = rate * prediction_sec
                preview_q[joint_index] = float(
                    self.hand_q[joint_index]
                    + np.clip(
                        preview_target - self.hand_q[joint_index],
                        -preview_step,
                        preview_step,
                    )
                )
                predicted_segment_cache[finger] = np.concatenate(
                    (
                        predicted_segment_cache[finger],
                        finger_capsule_segments(preview_q, finger)[
                            first_segment:
                        ],
                    ),
                    axis=0,
                )
                avoidance_sources.add(int(finger))

        self.inactive_collision_min_clearance_m = (
            min(observed_clearances) if observed_clearances else -1.0
        )
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
            if (
                finger == PINKY_FINGER_ID
                and self.pinky_special_hold_pose is not None
            ):
                target = self.pinky_special_hold_pose
            else:
                # Every unused finger waits at the complete selected
                # pre-grasp pose. It is therefore ready to join the grasp
                # without a separate PD preparation stage.
                target = pre_grasp_pose[idxs].copy()
                avoidance_joint = int(
                    FINGER_AVOIDANCE_JOINT_LOCAL_INDEX[finger]
                )
                target[avoidance_joint] += avoidance_offsets[
                    int(finger) - 1
                ]

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
                and self.pinky_special_hold_pose is None
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
        attach_forces = {}
        self.grasp_type7_transition_attach_tau_max = 0.0
        if not self._is_grasp_type7_attach_phase():
            return tau, attach_forces

        finger = self.grasp_type7_transition_finger_id
        if finger is None:
            return tau, attach_forces

        finger = int(finger)
        idxs = np.asarray(FINGER_JOINT_INDEX[finger], dtype=int)
        pos = tip_position(self.hand_q, finger)
        target_cv = self._grasp_type7_thumb_attach_centroid() if finger == G7_THUMB_TRANSITION_FINGER_ID else cv
        diff = target_cv - pos
        dist = np.linalg.norm(diff)
        if dist < 1e-9:
            return tau, attach_forces

        fhat = self.cfg.groped_force_direction_sign * diff / dist
        force = self._grasp_type7_attach_force(finger) * fhat
        attach_forces[finger] = force.copy()
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
        return tau, attach_forces

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
        if self.deferred_finger_count is not None:
            self.deferred_finger_count = None
            self.deferred_finger_count_at = None

    def apply_grasp_type(self, requested_count: int, now: float, *, internal: bool = False) -> None:
        if requested_count < -1 or requested_count > 7:
            raise ValueError("grasp_type must be one of -1, 0, 1, 2, 3, 4, 5, 6, 7")

        self.cancel_relative_rotation()
        self.cancel_relative_translation()
        self._reset_regular_force_balance_state()

        if not internal:
            self._cancel_deferred_for_external_command()

        command_count = requested_count

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
        rotation_enabled = False
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
            regular_grasp = 1 <= self.active_finger_count <= 5
            alpha_distribution_mode = (
                ALPHA_DISTRIBUTION_THUMB_DISTANCE_PROPORTIONAL
                if regular_grasp
                else ALPHA_DISTRIBUTION_LEGACY
            )
            if regular_grasp:
                effective_thumb_centroid_bias = 0.0
                relative_rotation_force_balance_blend = 1.0
            else:
                effective_thumb_centroid_bias = float(
                    self.cfg.thumb_centroid_bias
                )

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
                        rotation_enabled=rotation_enabled,
                        rotation_center=rotation_center_ref,
                        center_hold_target=rotation_center_ref,
                        center_hold_enabled=center_hold_enabled,
                        alpha_distribution_mode=alpha_distribution_mode,
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
                generic_rotation_forces = self._calc_relative_rotation_forces(
                    fingertip_positions,
                    now,
                    qdot,
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
                    )
                )
                if self.relative_translation_phase in {
                    "translating",
                    "translation_reached",
                }:
                    grasp_tau_without_translation = grasp_tau.copy()
                    (
                        grasp_tau,
                        relative_translation_joint_error,
                        relative_translation_position_torques,
                        relative_translation_nullspace_grasp_torques,
                        relative_translation_dls_sigma_min,
                        relative_translation_dls_condition,
                    ) = self._calc_relative_translation_joint_control(
                        base_total_forces=base_total_forces,
                        base_grasp_tau=grasp_tau_without_translation,
                        qdot=qdot,
                    )
                    translation_torques = (
                        grasp_tau - grasp_tau_without_translation
                    )
                    if self.relative_translation_phase in {
                        "translating",
                        "translation_reached",
                    }:
                        relative_translation_control_mode = (
                            "axis_centroid_dls_nullspace"
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

            g7_transition_pd = np.zeros(JOINT_COUNT, dtype=np.float64)
            g7_transition_attach_tau = np.zeros(JOINT_COUNT, dtype=np.float64)
            g7_transition_attach_forces = {}
            inactive_control_fingers = list(self.use_fingers)
            if (
                self.active_finger_count == PINKY_SPECIAL_COMMAND
                and self._is_grasp_type7_transition_phase()
            ):
                if self.grasp_type7_transition_finger_id is not None:
                    inactive_control_fingers.append(self.grasp_type7_transition_finger_id)
                g7_transition_pd = self._calc_grasp_type7_transition_pd(qdot)
                (
                    g7_transition_attach_tau,
                    g7_transition_attach_forces,
                ) = self._calc_grasp_type7_transition_attach_tau(cv)
                for finger, force in g7_transition_attach_forces.items():
                    zero = np.zeros(3, dtype=np.float64)
                    grasp_forces[finger] = (
                        grasp_forces.get(finger, zero) + force
                    )
                    total_forces[finger] = (
                        total_forces.get(finger, zero) + force
                    )

            inactive_pd = self._inactive_pre_grasp_pd(
                inactive_control_fingers,
                qdot,
                collision_avoidance_enabled=regular_grasp,
            )
            tau = (
                grasp_tau
                + inactive_pd
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
            rotation_enabled=bool(rotation_enabled),
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
            g7_phase=str(self.grasp_type7_phase),
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
