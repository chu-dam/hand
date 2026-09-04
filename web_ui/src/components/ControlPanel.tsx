import { useState } from "react";

import { rotateVectorToWorld } from "../ros/frames";
import type { GraspDebugMessage, HandSide, Point3, RotationMatrix3 } from "../ros/types";

const GRASP_OPTIONS = [
  { value: 1, label: "Thumb + Index", short: "2F-I" },
  { value: 2, label: "Thumb + Middle", short: "2F-M" },
  { value: 3, label: "Three finger", short: "3F" },
  { value: 4, label: "Four finger", short: "4F" },
  { value: 5, label: "Five finger", short: "5F" },
  { value: 6, label: "Envelop", short: "ENV" },
];

const POSE_OPTIONS = [
  { value: 1, label: "Normal" },
  { value: 2, label: "Pre-grasp" },
  { value: 3, label: "Compact" },
  { value: 4, label: "Card" },
];

const RIGHT_POSE_OPTIONS = [
  ...POSE_OPTIONS,
  { value: 5, label: "Pre-rotation" },
  { value: 6, label: "Pre-rotation (Blind Grasping)" },
];

const MANIPULATION_GRASP_TYPES = new Set([1, 2, 3, 4, 5]);
const TASK_SPACE_DIRECTIONS = ["+X", "+Y", "+Z", "-X", "-Y", "-Z"] as const;
type TaskSpaceDirection = typeof TASK_SPACE_DIRECTIONS[number];

const TASK_SPACE_UNIT_VECTORS: Record<TaskSpaceDirection, Point3> = {
  "+X": { x: 1, y: 0, z: 0 },
  "-X": { x: -1, y: 0, z: 0 },
  "+Y": { x: 0, y: 1, z: 0 },
  "-Y": { x: 0, y: -1, z: 0 },
  "+Z": { x: 0, y: 0, z: 1 },
  "-Z": { x: 0, y: 0, z: -1 },
};

function pointMillimeters(point: Point3 | undefined): string {
  if (!point) return "—";
  return [point.x, point.y, point.z]
    .map((value) => `${value >= 0 ? "+" : ""}${(1_000 * value).toFixed(2)}`)
    .join(", ");
}

function vectorNewtons(vector: Point3 | undefined): string {
  if (!vector) return "—";
  return [vector.x, vector.y, vector.z]
    .map((value) => `${value >= 0 ? "+" : ""}${value.toFixed(3)}`)
    .join(", ");
}

function maxAbsTorque(values: number[] | undefined): string {
  if (!Array.isArray(values) || values.length === 0) return "—";
  const finite = values.filter(Number.isFinite).map(Math.abs);
  if (finite.length === 0) return "—";
  return `${Math.max(...finite).toFixed(4)} N·m`;
}

const IDENTITY_MATRIX = ["1", "0", "0", "0", "1", "0", "0", "0", "1"];
const ROTATION_TOLERANCE = 0.03;

interface MatrixResult {
  values?: number[];
  error?: string;
}

function validateRotationMatrix(input: string[]): MatrixResult {
  if (input.length !== 9 || input.some((value) => value.trim() === "")) {
    return { error: "회전행렬의 9개 칸을 모두 입력해야 합니다." };
  }

  const values = input.map(Number);
  if (values.some((value) => !Number.isFinite(value))) {
    return { error: "회전행렬에는 유한한 숫자 9개가 필요합니다." };
  }

  const rowDot = (rowA: number, rowB: number) => (
    values[rowA * 3] * values[rowB * 3]
    + values[rowA * 3 + 1] * values[rowB * 3 + 1]
    + values[rowA * 3 + 2] * values[rowB * 3 + 2]
  );
  const orthogonalityErrors = [];
  for (let rowA = 0; rowA < 3; rowA += 1) {
    for (let rowB = 0; rowB < 3; rowB += 1) {
      orthogonalityErrors.push(Math.abs(rowDot(rowA, rowB) - (rowA === rowB ? 1 : 0)));
    }
  }

  const determinant = (
    values[0] * (values[4] * values[8] - values[5] * values[7])
    - values[1] * (values[3] * values[8] - values[5] * values[6])
    + values[2] * (values[3] * values[7] - values[4] * values[6])
  );
  if (
    Math.max(...orthogonalityErrors) > ROTATION_TOLERANCE
    || Math.abs(determinant - 1) > ROTATION_TOLERANCE
  ) {
    return {
      error: `유효한 회전행렬이 아닙니다. RᵀR≈I, det(R)≈1이어야 합니다 (현재 det=${determinant.toFixed(3)}).`,
    };
  }

  return { values };
}

interface ControlPanelProps {
  handSide: HandSide;
  connected: boolean;
  ready: boolean;
  debug: GraspDebugMessage | null;
  onGrasp: (value: number) => boolean;
  onPose: (value: number) => boolean;
  onAlpha: (value: number) => boolean;
  onTeaching: (value: boolean) => boolean;
  onRotationMatrix: (value: number[]) => boolean;
  onRelativeTranslation: (deltaWorldMeters: Point3) => boolean;
  onRelativeRotation: (degrees: number) => boolean;
  onContinuousRotation: (enable: boolean) => boolean;
  onBlindDirectionToggle: () => boolean;
  handToWorldRotation: RotationMatrix3;
  onNotice: (message: string, tone?: "ok" | "warning" | "error") => void;
}

export function ControlPanel({
  handSide,
  connected,
  ready,
  debug,
  onGrasp,
  onPose,
  onAlpha,
  onTeaching,
  onRotationMatrix,
  onRelativeTranslation,
  onRelativeRotation,
  onContinuousRotation,
  onBlindDirectionToggle,
  handToWorldRotation,
  onNotice,
}: ControlPanelProps) {
  const [alphaInput, setAlphaInput] = useState("3");
  const [translationMmInput, setTranslationMmInput] = useState("5");
  const [taskSpaceDirection, setTaskSpaceDirection] = useState<TaskSpaceDirection>("+X");
  const [rotationDegreesInput, setRotationDegreesInput] = useState("5");
  const [matrix, setMatrix] = useState([...IDENTITY_MATRIX]);
  const teaching = Boolean(debug?.teaching_mode);
  const continuousRotationActive = (
    (debug?.controller_phase?.startsWith("continuous_") === true
      || debug?.controller_phase?.startsWith("blind_") === true)
    && debug.controller_phase !== "continuous_error"
  );
  const commandDisabled = !ready || teaching;
  const cardGraspAvailable = (
    ready
    && !teaching
    && debug?.controller_state === "PRE_GRASP_POSE"
    && debug?.pose_type === 4
  );
  const manipulationSectionActive = (
    ready
    && !teaching
    && debug?.controller_state === "GROPED_GRASP"
    && MANIPULATION_GRASP_TYPES.has(debug?.grasp_type ?? -1)
    && debug?.controller_phase !== "force_balance_error"
    && !continuousRotationActive
  );
  const taskSpaceSectionActive = manipulationSectionActive;
  const rotationSectionActive = manipulationSectionActive;
  const matrixAllowed = ready && !teaching && debug?.controller_state === "NORMAL_POSE";
  const parsedAlpha = Number(alphaInput);
  const parsedTranslationMm = Number(translationMmInput);
  const translationMmValid = (
    translationMmInput.trim() !== ""
    && Number.isFinite(parsedTranslationMm)
    && parsedTranslationMm > 0
    && parsedTranslationMm <= 20
  );
  const parsedRotationDegrees = Number(rotationDegreesInput);
  const rotationDegreesValid = (
    rotationDegreesInput.trim() !== ""
    && Number.isFinite(parsedRotationDegrees)
  );
  const rotationCommandEnabled = rotationSectionActive
    && rotationDegreesValid
    && parsedRotationDegrees !== 0;
  const continuousRotationAvailable = (
    handSide === "right"
    && ready
    && !teaching
    && (
      continuousRotationActive
      || (
        debug?.controller_state === "PRE_GRASP_POSE"
        && debug?.pose_type === 5
      )
    )
  );
  const blindGraspContinuousRotationAvailable = (
    handSide === "right"
    && ready
    && !teaching
    && debug?.controller_state === "PRE_GRASP_POSE"
    && debug?.pose_type === 6
  );
  const blindDirectionAvailable = (
    handSide === "right"
    && ready
    && !teaching
    && debug?.pose_type === 6
    && (
      continuousRotationActive
      || debug?.controller_state === "GROPED_GRASP"
    )
  );
  const rotationDirection = !rotationDegreesValid
    ? "SET ANGLE"
    : parsedRotationDegrees > 0
      ? "CCW"
      : parsedRotationDegrees < 0
        ? "CW"
        : "NO ROTATION";
  const rotationDirectionClass = !rotationDegreesValid
    ? "invalid"
    : parsedRotationDegrees > 0
      ? "ccw"
      : parsedRotationDegrees < 0
        ? "cw"
        : "zero";
  const signedRotationDegrees = rotationDegreesValid
    ? `${parsedRotationDegrees > 0 ? "+" : ""}${parsedRotationDegrees}°`
    : "—";
  const translationPhase = debug?.relative_translation_phase ?? "idle";
  const rotationPhase = debug?.relative_rotation_phase ?? "idle";
  const rotationTargetDegrees = Number(debug?.relative_rotation_target_rad ?? 0) * 180 / Math.PI;
  const rotationCurrentDegrees = Number(debug?.relative_rotation_current_rad ?? 0) * 180 / Math.PI;
  const rotationErrorDegrees = Number(debug?.relative_rotation_error_rad ?? 0) * 180 / Math.PI;
  const translationTargetReady = translationPhase !== "idle";
  const translationErrorWorld = debug?.relative_translation_error
    ? rotateVectorToWorld(debug.relative_translation_error, handToWorldRotation)
    : undefined;
  const translationForceWorld = debug?.relative_translation_command_force
    ? rotateVectorToWorld(debug.relative_translation_command_force, handToWorldRotation)
    : undefined;
  const sliderAlpha = Number.isFinite(parsedAlpha)
    ? Math.min(10, Math.max(0, parsedAlpha))
    : 0;

  const report = (sent: boolean, requestMessage: string) => {
    onNotice(
      sent
        ? `${requestMessage} Debug 반영값을 확인하세요.`
        : "ROS 연결 또는 실시간 데이터가 없어 명령을 보내지 못했습니다.",
      sent ? "ok" : "error",
    );
  };

  const applyAlpha = () => {
    if (alphaInput.trim() === "" || !Number.isFinite(parsedAlpha) || parsedAlpha < 0 || parsedAlpha > 10) {
      onNotice("Alpha1은 0~10 범위의 유한한 값이어야 합니다.", "error");
      return;
    }
    report(onAlpha(parsedAlpha), `Alpha1 ${parsedAlpha.toFixed(2)} 요청을 전송했습니다.`);
  };

  const applyMatrix = () => {
    if (!matrixAllowed) {
      onNotice("Hand orientation은 실시간 데이터가 정상이고 NORMAL_POSE일 때만 적용할 수 있습니다.", "error");
      return;
    }
    const result = validateRotationMatrix(matrix);
    if (!result.values) {
      onNotice(result.error || "회전행렬을 확인해 주세요.", "error");
      return;
    }
    report(onRotationMatrix(result.values), "Hand orientation 행렬 요청을 전송했습니다.");
  };

  const prepareRelativeRotation = () => {
    if (!rotationSectionActive) {
      onNotice("Relative rotation은 활성 grasp type 1~5에서만 요청할 수 있습니다.", "error");
      return;
    }
    if (!rotationDegreesValid || parsedRotationDegrees === 0) {
      onNotice("Relative angle은 0이 아닌 유한한 값이어야 합니다.", "error");
      return;
    }
    report(
      onRelativeRotation(parsedRotationDegrees),
      `${signedRotationDegrees} relative rotation target 요청을 전송했습니다.`,
    );
  };

  const prepareRelativeTranslation = () => {
    if (!taskSpaceSectionActive) {
      onNotice("Translation은 활성 grasp type 1~5에서만 요청할 수 있습니다.", "error");
      return;
    }
    if (!translationMmValid) {
      onNotice("Relative distance는 0보다 크고 20 mm 이하여야 합니다.", "error");
      return;
    }
    const unit = TASK_SPACE_UNIT_VECTORS[taskSpaceDirection];
    const distanceMeters = parsedTranslationMm / 1_000;
    const deltaWorld = {
      x: distanceMeters * unit.x,
      y: distanceMeters * unit.y,
      z: distanceMeters * unit.z,
    };
    report(
      onRelativeTranslation(deltaWorld),
      `World ${taskSpaceDirection} ${parsedTranslationMm} mm target 요청을 전송했습니다.`,
    );
  };

  return (
    <aside className="control-column">
      <section className="panel control-panel">
        <div className="panel-head compact-head">
          <div>
            <p className="section-kicker">HIGH-LEVEL CONTROL</p>
            <h2>DG5F-S Commands</h2>
          </div>
          <span className={`command-ready ${ready ? "ready" : connected ? "waiting" : "offline"}`}>
            {ready ? "READY" : connected ? "WAIT DATA" : "OFFLINE"}
          </span>
        </div>

        <div className="control-section">
          <div className="control-section-title">
            <div><span>01</span><strong>Teaching Mode</strong></div>
            <small>{teaching ? "manual guidance active" : "controller active"}</small>
          </div>
          <div className="segmented two-column">
            <button
              className={teaching ? "active warning-active" : ""}
              disabled={!ready}
              onClick={() => report(onTeaching(true), "Teaching Mode ON 요청을 전송했습니다.")}
            >
              Teaching ON
            </button>
            <button
              className={!teaching && debug ? "active" : ""}
              disabled={!ready}
              onClick={() => report(onTeaching(false), "Teaching Mode OFF 요청을 전송했습니다.")}
            >
              Teaching OFF
            </button>
          </div>
          {teaching && (
            <p className="inline-warning">Teaching 중에는 Pose와 Grasp 명령이 잠깁니다.</p>
          )}
        </div>

        <div className="control-section">
          <div className="control-section-title">
            <div><span>02</span><strong>Pose</strong></div>
            <small>current {debug?.pose_type ?? "—"}</small>
          </div>
          <div className={`segmented pose-grid ${handSide === "right" ? "right-pose-grid" : ""}`}>
            {(handSide === "right" ? RIGHT_POSE_OPTIONS : POSE_OPTIONS).map((option) => (
              <button
                key={option.value}
                className={debug?.pose_type === option.value ? "active" : ""}
                disabled={commandDisabled}
                onClick={() => report(onPose(option.value), `Pose ${option.value} 요청을 전송했습니다.`)}
              >
                {option.label}
              </button>
            ))}
          </div>
          <button
            className="secondary-wide"
            disabled={commandDisabled}
            onClick={() => report(onGrasp(0), "선택된 Pre-grasp 이동 요청을 전송했습니다.")}
          >
            Move to selected pre-grasp
          </button>
        </div>

        <div className="control-section">
          <div className="control-section-title">
            <div><span>03</span><strong>Grasp Type</strong></div>
            <small>current {debug?.grasp_type ?? "—"}</small>
          </div>
          <div className="grasp-grid">
            {GRASP_OPTIONS.map((option) => (
              <button
                key={option.value}
                className={debug?.grasp_type === option.value ? "active" : ""}
                disabled={commandDisabled}
                title={option.label}
                onClick={() => report(onGrasp(option.value), `Grasp Type ${option.value} 요청을 전송했습니다.`)}
              >
                <strong>{option.short}</strong>
                <span>{option.label}</span>
              </button>
            ))}
            <button
              className={debug?.grasp_type === 7 ? "active" : ""}
              disabled={!cardGraspAvailable}
              title={cardGraspAvailable
                ? "Start Card grasp"
                : "Card grasp is available only in CARD pre-grasp"}
              onClick={() => report(onGrasp(7), "CARD grasp 요청을 전송했습니다.")}
            >
              <strong>CARD</strong>
              <span>Pre-grasp only</span>
            </button>
          </div>
        </div>

        <div className="control-section">
          <div className="control-section-title">
            <div><span>04</span><strong>Translation</strong></div>
            <small>2F-I · 2F-M · 3F · 4F · 5F only</small>
          </div>
          <div
            className={`rotation-slot task-space-slot ${taskSpaceSectionActive ? "active" : "locked"}`}
            aria-disabled={!taskSpaceSectionActive}
          >
            <div className="task-space-input-row">
              <label className="task-space-distance-field" htmlFor="task-space-distance-mm">
                <span>Relative distance from current pose</span>
                <div className="task-space-distance-input">
                  <input
                    id="task-space-distance-mm"
                    className="number-input"
                    type="number"
                    min="0.1"
                    max="20"
                    step="0.1"
                    inputMode="decimal"
                    value={translationMmInput}
                    disabled={!taskSpaceSectionActive}
                    aria-invalid={translationMmInput.trim() !== "" && !translationMmValid}
                    onChange={(event) => setTranslationMmInput(event.target.value)}
                  />
                  <span>mm</span>
                </div>
              </label>
              <div className="task-space-frame-preview">
                <span>Task frame</span>
                <strong>{taskSpaceSectionActive ? "WORLD XYZ" : "LOCKED"}</strong>
              </div>
            </div>

            <div className="task-space-direction-grid" aria-label="Task-space direction">
              {TASK_SPACE_DIRECTIONS.map((direction) => (
                <button
                  key={direction}
                  type="button"
                  className={taskSpaceDirection === direction ? "selected" : ""}
                  disabled={!taskSpaceSectionActive || !translationMmValid}
                  aria-pressed={taskSpaceDirection === direction}
                  onClick={() => setTaskSpaceDirection(direction)}
                >
                  {direction}
                </button>
              ))}
            </div>

            <div className="task-space-selection-row">
              <div>
                <span>Selected relative move</span>
                <strong>
                  {taskSpaceSectionActive && translationMmValid
                    ? `${taskSpaceDirection} · ${parsedTranslationMm} mm`
                    : "—"}
                </strong>
              </div>
              <button
                type="button"
                className="apply-button task-space-prepare-button"
                disabled={!taskSpaceSectionActive || !translationMmValid}
                onClick={prepareRelativeTranslation}
              >
                Move
              </button>
            </div>
            <div className={`task-space-debug-grid ${translationTargetReady ? "ready" : "idle"}`}>
              <div>
                <span>Current Cg · link_base [mm]</span>
                <strong>{pointMillimeters(debug?.geometric_centroid)}</strong>
              </div>
              <div>
                <span>Target Cg · link_base [mm]</span>
                <strong>{translationTargetReady
                  ? pointMillimeters(debug?.relative_translation_target_centroid)
                  : "—"}</strong>
              </div>
              <div>
                <span>Remaining · world XYZ [mm]</span>
                <strong>{translationTargetReady
                  ? pointMillimeters(translationErrorWorld)
                  : "—"}</strong>
              </div>
              <div>
                <span>Virtual task force · world XYZ [N]</span>
                <strong>{translationTargetReady
                  ? vectorNewtons(translationForceWorld)
                  : "—"}</strong>
              </div>
              <div>
                <span>Translation torque max · 20 joints</span>
                <strong>{translationTargetReady
                  ? maxAbsTorque(debug?.translation_torques)
                  : "—"}</strong>
              </div>
              <div>
                <span>Control mapping</span>
                <strong>{translationTargetReady ? "Jᵢᵀ(Fgrasp + Ftranslation)" : "—"}</strong>
              </div>
            </div>
            <p className="task-space-stage-note">
              {translationPhase === "translating"
                ? "TRANSLATING · Each fingertip tracks its captured Cartesian target through Jacobian transpose control."
                : translationPhase === "translation_reached"
                  ? "REACHED · Cartesian fingertip target hold is active."
                  : translationPhase === "translation_timeout"
                    ? "TIMEOUT · position control was removed. Check the grasp and retry with 1 mm."
                  : translationPhase === "translation_error"
                      ? "ERROR · translation control was removed because the Cartesian state was invalid."
                      : "Actual motion enabled. Start with 1 mm and keep RELEASE ready."}
            </p>
          </div>
        </div>

        <div className="control-section">
          <div className="control-section-title">
            <div><span>05</span><strong>Rotation</strong></div>
            <small>2F-I · 2F-M · 3F · 4F · 5F only</small>
          </div>
          <div
            className={`rotation-slot ${rotationSectionActive ? "active" : "locked"}`}
            aria-disabled={!rotationSectionActive}
          >
            <div className="rotation-input-row">
              <label className="rotation-angle-field" htmlFor="rotation-angle-degrees">
                <span>Relative angle from current pose</span>
                <div className="rotation-degree-input">
                  <input
                    id="rotation-angle-degrees"
                    className="number-input"
                    type="number"
                    step="1"
                    inputMode="decimal"
                    value={rotationDegreesInput}
                    disabled={!rotationSectionActive}
                    aria-invalid={rotationDegreesInput.trim() !== "" && !rotationDegreesValid}
                    onChange={(event) => setRotationDegreesInput(event.target.value)}
                  />
                  <span>deg</span>
                </div>
              </label>
              <div className="rotation-direction-preview">
                <span>Direction</span>
                <strong className={rotationSectionActive ? rotationDirectionClass : "locked"}>
                  {rotationSectionActive
                    ? `${rotationDirection} · ${signedRotationDegrees}`
                    : "LOCKED"}
                </strong>
              </div>
            </div>
            <p className="rotation-sign-hint">
              Positive (+): CCW · Negative (−): CW · palm-normal axis (link_base −X)
            </p>
            <div className={`rotation-debug-grid ${rotationPhase !== "idle" ? "ready" : "idle"}`}>
              <div>
                <span>Target</span>
                <strong>{rotationPhase !== "idle" ? `${rotationTargetDegrees.toFixed(2)}°` : "—"}</strong>
              </div>
              <div>
                <span>Estimated</span>
                <strong>{rotationPhase !== "idle" ? `${rotationCurrentDegrees.toFixed(2)}°` : "—"}</strong>
              </div>
              <div>
                <span>Remaining</span>
                <strong>{rotationPhase !== "idle" ? `${rotationErrorDegrees.toFixed(2)}°` : "—"}</strong>
              </div>
              <div>
                <span>Moment</span>
                <strong>{rotationPhase !== "idle"
                  ? `${Number(debug?.relative_rotation_command_moment ?? 0).toFixed(4)} N·m`
                  : "—"}</strong>
              </div>
            </div>
            <div className="rotation-action-row">
              <p className="rotation-stage-note">
                {rotationPhase === "rotating"
                  ? "ROTATING · tangential force and centroid hold are active."
                  : rotationPhase === "rotation_reached"
                    ? "REACHED · target contact angle is being held."
                    : rotationPhase === "rotation_timeout"
                      ? "TIMEOUT · rotation force was removed. Check contact slip and retry."
                      : rotationPhase === "rotation_error"
                        ? "ERROR · rotation force was removed because the contact geometry is invalid."
                        : rotationPhase === "force_balance_error"
                          ? "FORCE BALANCE ERROR · select the grasp type again before rotating."
                        : "Closed-loop estimate uses fingertip contacts, not an object-angle sensor."}
              </p>
              <button
                className="apply-button rotation-prepare-button"
                disabled={!rotationCommandEnabled}
                onClick={prepareRelativeRotation}
              >
                Rotate
              </button>
            </div>
          </div>
          <button
            className="secondary-wide apply-button"
            disabled={!continuousRotationAvailable}
            title={continuousRotationAvailable
              ? continuousRotationActive
                ? "Stop continuous rotation"
                : "Start continuous rotation"
              : "Available only in the right-hand Pre-rotation pose"}
            onClick={() => report(
              onContinuousRotation(!continuousRotationActive),
              continuousRotationActive
                ? "Continuous rotation 중지 요청을 전송했습니다."
                : "Continuous rotation 시작 요청을 전송했습니다.",
            )}
          >
            {continuousRotationActive
              ? "Stop continuous rotation"
              : "Continuous rotation"}
          </button>
          <div className="blind-regrasp-row">
            <button
              className="secondary-wide apply-button"
              disabled={!blindGraspContinuousRotationAvailable}
              title={blindGraspContinuousRotationAvailable
                ? "Run middle, index+ring, thumb, then pinky release sequence"
                : "Available only in the right-hand Pre-rotation (Blind Grasping) pose"}
              onClick={() => report(
                onContinuousRotation(true),
                "Blind regrasp sequence 시작 요청을 전송했습니다.",
              )}
            >
              Blind regrasp sequence
            </button>
            <button
              className="blind-direction-button"
              type="button"
              disabled={!blindDirectionAvailable}
              title="Toggle rotation direction"
              aria-label="Toggle rotation direction"
              onClick={() => onBlindDirectionToggle()}
            >
              ↔
            </button>
          </div>
        </div>

        <div className="control-section">
          <div className="control-section-title">
            <div><span>06</span><strong>Alpha1</strong></div>
            <small>thumb force magnitude reference · grasp types 1–5</small>
          </div>
          <div className="slider-row">
            <input
              type="range"
              min="0"
              max="10"
              step="0.1"
              value={sliderAlpha}
              disabled={commandDisabled || continuousRotationActive}
              onChange={(event) => setAlphaInput(event.target.value)}
            />
            <input
              className="number-input"
              type="number"
              min="0"
              max="10"
              step="0.1"
              value={alphaInput}
              disabled={commandDisabled || continuousRotationActive}
              onChange={(event) => setAlphaInput(event.target.value)}
            />
            <button
              className="apply-button"
              disabled={commandDisabled || continuousRotationActive}
              onClick={applyAlpha}
            >
              Apply
            </button>
          </div>
        </div>

        <details className="matrix-section">
          <summary>
            <span><b>07</b> Hand orientation</span>
            <small>gravity compensation · NORMAL_POSE only</small>
          </summary>
          <div className="matrix-grid">
            {matrix.map((value, index) => (
              <label key={index}>
                <span>R{Math.floor(index / 3) + 1}{(index % 3) + 1}</span>
                <input
                  type="number"
                  step="0.001"
                  value={value}
                  disabled={!connected}
                  onChange={(event) => {
                    const next = [...matrix];
                    next[index] = event.target.value;
                    setMatrix(next);
                  }}
                />
              </label>
            ))}
          </div>
          {!matrixAllowed && connected && (
            <p className="inline-warning">실시간 Debug와 NORMAL_POSE 상태에서만 행렬을 적용할 수 있습니다.</p>
          )}
          <div className="matrix-actions">
            <button
              className="ghost-button"
              disabled={!connected}
              onClick={() => setMatrix([...IDENTITY_MATRIX])}
            >
              Identity
            </button>
            <button className="apply-button" disabled={!matrixAllowed} onClick={applyMatrix}>Apply matrix</button>
          </div>
        </details>

        <div className="release-zone">
          <div>
            <strong>Release / Normal Pose</strong>
            <span>Emergency stop이 아닌 정상 자세 이동 명령입니다.</span>
          </div>
          <button
            disabled={commandDisabled}
            onClick={() => report(onGrasp(-1), "Release 요청을 전송했습니다.")}
          >
            RELEASE
          </button>
        </div>
      </section>
    </aside>
  );
}
