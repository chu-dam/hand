import { useState } from "react";

import type { GraspDebugMessage } from "../ros/types";

const GRASP_OPTIONS = [
  { value: 1, label: "Thumb + Index", short: "2F-I" },
  { value: 2, label: "Thumb + Middle", short: "2F-M" },
  { value: 3, label: "Three finger", short: "3F" },
  { value: 4, label: "Four finger", short: "4F" },
  { value: 5, label: "Five finger", short: "5F" },
  { value: 6, label: "Envelop", short: "ENV" },
  { value: 7, label: "Rotation", short: "ROT" },
];

const POSE_OPTIONS = [
  { value: 1, label: "Normal" },
  { value: 2, label: "Pre-grasp" },
  { value: 3, label: "Compact" },
];

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
  connected: boolean;
  ready: boolean;
  debug: GraspDebugMessage | null;
  onGrasp: (value: number) => boolean;
  onPose: (value: number) => boolean;
  onAlpha: (value: number) => boolean;
  onTeaching: (value: boolean) => boolean;
  onRotationMatrix: (value: number[]) => boolean;
  onNotice: (message: string, tone?: "ok" | "warning" | "error") => void;
}

export function ControlPanel({
  connected,
  ready,
  debug,
  onGrasp,
  onPose,
  onAlpha,
  onTeaching,
  onRotationMatrix,
  onNotice,
}: ControlPanelProps) {
  const [alphaInput, setAlphaInput] = useState("3");
  const [matrix, setMatrix] = useState([...IDENTITY_MATRIX]);
  const teaching = Boolean(debug?.teaching_mode);
  const commandDisabled = !ready || teaching;
  const matrixAllowed = ready && !teaching && debug?.controller_state === "NORMAL_POSE";
  const parsedAlpha = Number(alphaInput);
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
          <div className="segmented three-column">
            {POSE_OPTIONS.map((option) => (
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
          </div>
        </div>

        <div className="control-section">
          <div className="control-section-title">
            <div><span>04</span><strong>Alpha1</strong></div>
            <small>calculated force coefficient</small>
          </div>
          <div className="slider-row">
            <input
              type="range"
              min="0"
              max="10"
              step="0.1"
              value={sliderAlpha}
              disabled={commandDisabled}
              onChange={(event) => setAlphaInput(event.target.value)}
            />
            <input
              className="number-input"
              type="number"
              min="0"
              max="10"
              step="0.1"
              value={alphaInput}
              disabled={commandDisabled}
              onChange={(event) => setAlphaInput(event.target.value)}
            />
            <button className="apply-button" disabled={commandDisabled} onClick={applyAlpha}>Apply</button>
          </div>
        </div>

        <details className="matrix-section">
          <summary>
            <span><b>05</b> Hand orientation</span>
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
