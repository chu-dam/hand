from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional

import numpy as np

from dg5f_grasp_control.config import RuntimeConfig
from dg5f_grasp_control.control_utils import pose_pd
from dg5f_grasp_control.grasp_policy import (
    ALPHA_DISTRIBUTION_LEGACY,
    ALPHA_DISTRIBUTION_THUMB_DISTANCE_PROPORTIONAL,
    GraspPolicy,
    GraspPolicyResult,
)
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
    envelop_info: Dict[str, object] = field(default_factory=dict)
    g7_phase: str = "idle"
    relative_rotation_phase: str = "idle"
    relative_rotation_target_rad: float = 0.0
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

        # Generic grasp types (1..5) already use Cv == Cg and balanced grasp
        # forces. A relative command therefore only stores the signed target
        # for the future rotation stage; it applies no rotation force yet.
        self.relative_rotation_phase = "idle"
        self.relative_rotation_target_rad = 0.0
        self.relative_rotation_started_at = None
        # Relative task-space motion state. The grasp policy keeps the normal
        # holding force active while a Cartesian impedance tracks one stored
        # target for every active fingertip.
        self.relative_translation_phase = "idle"
        self.relative_translation_start_centroid = np.zeros(3, dtype=np.float64)
        self.relative_translation_target_centroid = np.zeros(3, dtype=np.float64)
        self.relative_translation_delta = np.zeros(3, dtype=np.float64)
        self.relative_translation_error = np.zeros(3, dtype=np.float64)
        self.relative_translation_centroid_velocity = np.zeros(3, dtype=np.float64)
        self.relative_translation_command_force = np.zeros(3, dtype=np.float64)
        self.relative_translation_start_fingertips = {}
        self.relative_translation_target_fingertips = {}
        self.relative_translation_last_fingertips = {}
        self.relative_translation_fingertip_velocities = {}
        self.relative_translation_contact_weights = {}
        self.relative_translation_max_fingertip_error = 0.0
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

    def cancel_relative_translation(self) -> None:
        """Cancel the stored Cartesian translation target and impedance."""

        self.relative_translation_phase = "idle"
        self.relative_translation_start_centroid[:] = 0.0
        self.relative_translation_target_centroid[:] = 0.0
        self.relative_translation_delta[:] = 0.0
        self.relative_translation_error[:] = 0.0
        self.relative_translation_centroid_velocity[:] = 0.0
        self.relative_translation_command_force[:] = 0.0
        self.relative_translation_start_fingertips = {}
        self.relative_translation_target_fingertips = {}
        self.relative_translation_last_fingertips = {}
        self.relative_translation_fingertip_velocities = {}
        self.relative_translation_contact_weights = {}
        self.relative_translation_max_fingertip_error = 0.0
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
        """Store a signed rotation target relative to the current object pose.

        Regular grasp types already use Cv == Cg and force-balanced
        coefficients, so the preparation is immediately ready. No tangential
        rotation force is generated by the current generic grasp path.
        """

        angle_rad = float(angle_rad)
        now = float(now)
        if not np.isfinite(angle_rad) or angle_rad == 0.0:
            raise ValueError("relative rotation angle must be finite and non-zero")
        if not np.isfinite(now):
            raise ValueError("now must be finite")
        if self.state != "GROPED_GRASP" or self.active_finger_count not in range(1, 6):
            self._log(
                "[RELATIVE_ROTATION] ignored: requires active grasp_type 1..5 "
                f"(state={self.state}, grasp_type={self.active_finger_count})"
            )
            return False
        if (
            self.deferred_finger_count is not None
            or self.adding_finger_target_count is not None
            or bool(self.blending_fingers)
        ):
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

        self.cancel_relative_translation()
        self.relative_rotation_target_rad = angle_rad
        self.relative_rotation_phase = "rotation_ready"
        self.relative_rotation_started_at = None
        self._log(
            "[RELATIVE_ROTATION] centroid and force distribution ready "
            f"target_delta_deg={np.degrees(angle_rad):.3f}"
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
        if (
            self.deferred_finger_count is not None
            or self.adding_finger_target_count is not None
            or bool(self.blending_fingers)
        ):
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
        self.relative_translation_max_fingertip_error = distance
        self.relative_translation_reference_progress = 0.0
        self.relative_translation_control_axis_error = 0.0
        self.relative_translation_control_axis_drive_force = 0.0
        self.relative_translation_started_at = now
        self.relative_translation_last_time = now
        self.relative_translation_last_centroid = start.copy()
        self.relative_translation_reached_since = None
        self.relative_translation_phase = "translating"
        self._log(
            "[RELATIVE_TRANSLATION] hybrid grasp/Cartesian fingertip targets started "
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

    def _calc_relative_translation_forces(
        self,
        cg: np.ndarray,
        tip_positions: Dict[int, np.ndarray],
        now: float,
    ) -> Dict[int, np.ndarray]:
        """Track independent Cartesian fingertip targets while grasping."""

        zero = self._zero_translation_forces(tip_positions)
        self.relative_translation_command_force[:] = 0.0
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
        self.relative_translation_max_fingertip_error = max(
            float(np.linalg.norm(error))
            for error in fingertip_errors.values()
        )

        translation_distance = float(np.linalg.norm(self.relative_translation_delta))
        if translation_distance <= 1e-12:
            self.relative_translation_phase = "translation_error"
            self._log("[RELATIVE_TRANSLATION] stopped: zero translation direction")
            return zero
        translation_axis = self.relative_translation_delta / translation_distance
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
                "[RELATIVE_TRANSLATION] timeout; motion force removed "
                f"centroid_error_mm="
                f"{np.round(1000.0 * self.relative_translation_error, 3).tolist()}, "
                f"max_tip_error_mm="
                f"{1000.0 * self.relative_translation_max_fingertip_error:.3f}"
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
        position_error_norm = float(np.linalg.norm(self.relative_translation_error))
        maximum_fingertip_velocity = max(
            (
                float(np.linalg.norm(velocity))
                for velocity in self.relative_translation_fingertip_velocities.values()
            ),
            default=0.0,
        )
        velocity_norm = max(
            float(np.linalg.norm(self.relative_translation_centroid_velocity)),
            maximum_fingertip_velocity,
        )
        inside_tolerance = (
            reference_progress >= 1.0 - 1e-12
            and position_error_norm <= position_tolerance
            and self.relative_translation_max_fingertip_error <= position_tolerance
            and velocity_norm <= velocity_tolerance
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
                    f"{1000.0 * self.relative_translation_max_fingertip_error:.3f}"
                )
        else:
            self.relative_translation_reached_since = None
            if (
                self.relative_translation_phase == "translation_reached"
                and max(
                    position_error_norm,
                    self.relative_translation_max_fingertip_error,
                )
                > 1.5 * max(position_tolerance, 1e-9)
            ):
                self.relative_translation_phase = "translating"

        kp = float(self.cfg.relative_translation_kp)
        kd = float(self.cfg.relative_translation_kd)
        hold_kp = float(self.cfg.relative_translation_hold_kp)
        hold_kd = float(self.cfg.relative_translation_hold_kd)
        shape_kp = float(self.cfg.relative_translation_shape_kp)
        shape_kd = float(self.cfg.relative_translation_shape_kd)
        cross_axis_deadband = float(
            self.cfg.relative_translation_cross_axis_deadband_m
        )
        gains = (kp, kd, hold_kp, hold_kd, shape_kp, shape_kd)
        if (
            not all(np.isfinite(gain) and gain >= 0.0 for gain in gains)
            or not np.isfinite(cross_axis_deadband)
            or cross_axis_deadband < 0.0
        ):
            self.relative_translation_phase = "translation_error"
            self._log("[RELATIVE_TRANSLATION] stopped: invalid Cartesian gains")
            return zero

        centroid_velocity = self.relative_translation_centroid_velocity
        axis_error = (
            translation_axis
            * float(np.dot(translation_axis, control_centroid_error))
        )
        cross_axis_error = control_centroid_error - axis_error
        cross_axis_error_norm = float(np.linalg.norm(cross_axis_error))
        if cross_axis_error_norm <= cross_axis_deadband:
            cross_axis_error[:] = 0.0
        elif cross_axis_deadband > 0.0:
            cross_axis_error *= (
                1.0 - cross_axis_deadband / cross_axis_error_norm
            )
        axis_velocity = (
            translation_axis
            * float(np.dot(translation_axis, centroid_velocity))
        )
        cross_axis_velocity = centroid_velocity - axis_velocity
        object_force = (
            kp * axis_error
            - kd * axis_velocity
            + hold_kp * cross_axis_error
            - hold_kd * cross_axis_velocity
        )
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
        self.relative_translation_command_force = sum(
            forces.values(),
            np.zeros(3, dtype=np.float64),
        )
        return forces

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
        if self.deferred_finger_count is not None and not self.skip_add_pre_grasp_once:
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
        relative_translation_torque_target = 0.0
        relative_translation_force_scale = 0.0
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
                    self.relative_rotation_phase = "force_balance_error"
                    self.relative_rotation_target_rad = 0.0
                    self.relative_rotation_started_at = None
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
                translation_forces.update(
                    self._calc_relative_translation_forces(
                        cg,
                        fingertip_positions,
                        now,
                    )
                )
                if any(
                    np.linalg.norm(force) > 0.0
                    for force in translation_forces.values()
                ):
                    grasp_tau_without_translation = grasp_tau.copy()
                    relative_translation_force_scale = 1.0

                    def apply_translation_forces():
                        combined_forces = _copy_finger_vectors(base_total_forces)
                        for active_finger, active_force in translation_forces.items():
                            combined_forces[active_finger] += active_force
                        combined_tau = self.policy.calc_tau_from_total_forces(
                            self.hand_q,
                            combined_forces,
                        )
                        return combined_forces, combined_tau

                    total_forces, grasp_tau = apply_translation_forces()
                    # This is the exact incremental controller torque that
                    # survives the same clipping as the combined grasp command.
                    translation_torques = grasp_tau - grasp_tau_without_translation

                    if bool(
                        self.cfg.relative_translation_torque_normalization_enable
                    ):
                        torque_gain = max(
                            0.0,
                            float(
                                self.cfg.relative_translation_torque_gain_nm_per_m
                            ),
                        )
                        torque_limit = max(
                            0.0,
                            float(self.cfg.relative_translation_torque_limit),
                        )
                        position_gain = max(
                            0.0,
                            float(self.cfg.relative_translation_kp),
                        )
                        damped_equivalent_error = (
                            abs(
                                self.relative_translation_control_axis_drive_force
                            )
                            / position_gain
                            if position_gain > 1e-12
                            else 0.0
                        )
                        translation_axis = self.relative_translation_delta.copy()
                        translation_axis /= max(
                            float(np.linalg.norm(translation_axis)),
                            1e-12,
                        )
                        axis_multipliers = np.array(
                            [
                                self.cfg.relative_translation_torque_axis_multiplier_x,
                                self.cfg.relative_translation_torque_axis_multiplier_y,
                                self.cfg.relative_translation_torque_axis_multiplier_z,
                            ],
                            dtype=np.float64,
                        )
                        if (
                            not np.all(np.isfinite(axis_multipliers))
                            or np.any(axis_multipliers <= 0.0)
                        ):
                            axis_multipliers = np.ones(3, dtype=np.float64)
                        direction_multiplier = float(
                            np.linalg.norm(axis_multipliers * translation_axis)
                        )
                        relative_translation_torque_target = min(
                            torque_gain
                            * direction_multiplier
                            * damped_equivalent_error,
                            torque_limit,
                        )

                        def axis_translation_torques():
                            axis_forces = _copy_finger_vectors(base_total_forces)
                            for active_finger, active_force in translation_forces.items():
                                axis_forces[active_finger] += (
                                    translation_axis
                                    * float(np.dot(translation_axis, active_force))
                                )
                            axis_tau = self.policy.calc_tau_from_total_forces(
                                self.hand_q,
                                axis_forces,
                            )
                            return axis_tau - grasp_tau_without_translation

                        def axis_scaled_forces(axis_scale):
                            proposed = {}
                            for active_finger, active_force in translation_forces.items():
                                parallel = (
                                    translation_axis
                                    * float(np.dot(translation_axis, active_force))
                                )
                                proposed[active_finger] = (
                                    active_force
                                    + (axis_scale - 1.0) * parallel
                                )
                            return proposed

                        force_limit = max(
                            0.0,
                            float(self.cfg.relative_translation_force_limit),
                        )
                        per_finger_limit = max(
                            0.0,
                            float(
                                self.cfg.relative_translation_per_finger_force_limit
                            ),
                        )

                        def forces_within_limits(proposed):
                            resultant = sum(
                                proposed.values(),
                                np.zeros(3, dtype=np.float64),
                            )
                            if (
                                force_limit > 0.0
                                and np.linalg.norm(resultant) > force_limit + 1e-12
                            ):
                                return False
                            if per_finger_limit > 0.0 and any(
                                np.linalg.norm(force)
                                > per_finger_limit + 1e-12
                                for force in proposed.values()
                            ):
                                return False
                            return True

                        # Two passes account for the fact that combined grasp
                        # torque clipping makes the incremental J^T F mapping
                        # mildly nonlinear. Only the commanded-axis component
                        # is boosted, so cross-axis hold and shape damping are
                        # not amplified into an oscillation.
                        for _ in range(2):
                            motion_axis_torques = axis_translation_torques()
                            torque_max = float(
                                np.max(np.abs(motion_axis_torques))
                            )
                            if (
                                torque_max <= 1e-12
                                or torque_max
                                >= relative_translation_torque_target - 1e-9
                            ):
                                break

                            scale = (
                                relative_translation_torque_target / torque_max
                            )
                            if scale <= 1.0 + 1e-9:
                                break

                            proposed = axis_scaled_forces(scale)
                            if not forces_within_limits(proposed):
                                lower = 1.0
                                upper = scale
                                for _ in range(24):
                                    candidate = 0.5 * (lower + upper)
                                    candidate_forces = axis_scaled_forces(candidate)
                                    if forces_within_limits(candidate_forces):
                                        lower = candidate
                                    else:
                                        upper = candidate
                                scale = lower
                                proposed = axis_scaled_forces(scale)
                            if scale <= 1.0 + 1e-9:
                                break

                            translation_forces = proposed
                            relative_translation_force_scale *= scale
                            self.relative_translation_command_force = sum(
                                translation_forces.values(),
                                np.zeros(3, dtype=np.float64),
                            )
                            total_forces, grasp_tau = apply_translation_forces()
                            translation_torques = (
                                grasp_tau - grasp_tau_without_translation
                            )

            blend_pd = np.zeros(JOINT_COUNT, dtype=np.float64)
            if (
                not using_force_balance_fallback
                and self.blending_fingers
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
                    translation_torques[idxs] *= blend
                    grasp_forces[finger] *= blend
                    translation_forces[finger] *= blend
                    rotation_forces[finger] *= blend
                    center_hold_forces[finger] *= blend
                    collision_forces[finger] *= blend
                    total_forces[finger] *= blend
                    blend_pd[idxs] = (1.0 - blend) * pd
                if blend_raw >= 1.0:
                    self.blending_fingers = []
                    self.blend_started_at = None
                    self.blend_until = None

            if (
                regular_grasp
                and not using_force_balance_fallback
                and not self.blending_fingers
            ):
                # Keep the fail-closed cache free of manipulation force. If a
                # later balance solve fails, translation must disappear
                # immediately while only the stable grasp command fades out.
                cached_total_forces = _copy_finger_vectors(total_forces)
                for finger, force in translation_forces.items():
                    cached_total_forces[finger] -= force
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
                    grasp_forces=_copy_finger_vectors(grasp_forces),
                    rotation_forces=_copy_finger_vectors(rotation_forces),
                    center_hold_forces=_copy_finger_vectors(
                        center_hold_forces
                    ),
                    collision_forces=_copy_finger_vectors(collision_forces),
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
            envelop_info=dict(self.envelop_last_info),
            g7_phase=str(self.grasp_type7_phase),
            relative_rotation_phase=str(self.relative_rotation_phase),
            relative_rotation_target_rad=float(self.relative_rotation_target_rad),
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
            effective_thumb_centroid_bias=float(effective_thumb_centroid_bias),
            relative_rotation_force_balance_blend=float(
                relative_rotation_force_balance_blend
            ),
        )
