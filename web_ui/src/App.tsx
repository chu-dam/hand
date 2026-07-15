import { useEffect, useState } from "react";

import { ControlPanel } from "./components/ControlPanel";
import { DebugReadout } from "./components/DebugReadout";
import { ForceHistoryPanel } from "./components/ForceHistoryPanel";
import { HandScene3D } from "./components/HandScene3D";
import { JointTable } from "./components/JointTable";
import { StatusPill } from "./components/StatusPill";
import { defaultRosbridgeUrl, useRosBridge } from "./ros/useRosBridge";

type NoticeTone = "ok" | "warning" | "error";

interface Notice {
  message: string;
  tone: NoticeTone;
}

function ageSeconds(timestamp: number | null, now: number): number | null {
  return timestamp === null ? null : Math.max(0, (now - timestamp) / 1_000);
}

function fmtAge(age: number | null) {
  return age === null ? "—" : `${age.toFixed(age < 1 ? 2 : 1)}s`;
}

function pointText(point: { x: number; y: number; z: number } | undefined) {
  if (!point) return "—";
  return `${point.x.toFixed(3)}, ${point.y.toFixed(3)}, ${point.z.toFixed(3)}`;
}

export function App() {
  const [rosbridgeUrl, setRosbridgeUrl] = useState(defaultRosbridgeUrl);
  const [urlDraft, setUrlDraft] = useState(rosbridgeUrl);
  const [now, setNow] = useState(Date.now());
  const [notice, setNotice] = useState<Notice>({
    message: "웹 UI가 준비되었습니다. rosbridge 연결을 기다리는 중입니다.",
    tone: "warning",
  });

  const ros = useRosBridge(rosbridgeUrl);
  const connected = ros.status === "connected";
  const jointAge = ageSeconds(ros.lastJointAt, now);
  const debugAge = ageSeconds(ros.lastDebugAt, now);
  const rotationAge = ageSeconds(ros.lastRotationAt, now);
  const handConnected = connected && jointAge !== null && jointAge < 1;
  const debugLive = connected && debugAge !== null && debugAge < 1;
  const controlsReady = connected && handConnected && debugLive;

  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 250);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    if (controlsReady) {
      setNotice({ message: "JointState와 GraspDebug가 정상입니다. 제어 명령을 사용할 수 있습니다.", tone: "ok" });
    } else if (connected) {
      setNotice({ message: "rosbridge 연결 완료 · JointState와 GraspDebug를 기다리는 중입니다.", tone: "warning" });
    } else if (ros.status === "error") {
      setNotice({ message: ros.error || "rosbridge 연결 오류", tone: "error" });
    } else if (ros.status === "disconnected") {
      setNotice({ message: "rosbridge 연결이 끊겼습니다. 자동 재연결을 시도합니다.", tone: "warning" });
    }
  }, [connected, controlsReady, ros.error, ros.status]);

  const pushNotice = (message: string, tone: NoticeTone = "ok") => {
    setNotice({ message, tone });
  };

  const connectFromDraft = () => {
    const candidate = urlDraft.trim();
    try {
      const parsed = new URL(candidate);
      if (parsed.protocol !== "ws:" && parsed.protocol !== "wss:") throw new Error();
    } catch {
      pushNotice("올바른 ws:// 또는 wss:// 주소를 입력해 주세요.", "error");
      return;
    }

    if (candidate === rosbridgeUrl) {
      ros.reconnect();
    } else {
      setRosbridgeUrl(candidate);
    }
  };

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-block">
          <div className="brand-mark" aria-hidden="true">
            <span />
            <span />
            <span />
          </div>
          <div>
            <p className="eyebrow">DG5F-S · ROBOT HAND</p>
            <h1>Grasp Control Console</h1>
            <p className="subtitle">Calculated force visualization and high-level ROS 2 control</p>
          </div>
        </div>
        <div className="status-strip">
          <StatusPill
            label="ROS"
            value={connected ? "CONNECTED" : ros.status.toUpperCase()}
            tone={connected ? "ok" : ros.status === "error" ? "danger" : "wait"}
          />
          <StatusPill
            label="HAND"
            value={handConnected ? "LIVE" : "WAITING"}
            tone={handConnected ? "ok" : "wait"}
          />
          <StatusPill
            label="DEBUG"
            value={debugLive ? fmtAge(debugAge) : "WAITING"}
            tone={debugLive ? "ok" : "wait"}
          />
          <StatusPill
            label="STATE"
            value={ros.debug?.controller_state || "—"}
            tone="neutral"
          />
        </div>
      </header>

      <section className="connection-bar">
        <div className="connection-copy">
          <span className={`connection-light ${connected ? "online" : ""}`} />
          <div>
            <strong>ROSBridge WebSocket</strong>
            <small>
              JointState {fmtAge(jointAge)} · GraspDebug {fmtAge(debugAge)} · Orientation {rotationAge === null ? "DEFAULT I" : fmtAge(rotationAge)}
            </small>
          </div>
        </div>
        <div className="connection-form">
          <input
            aria-label="rosbridge websocket URL"
            value={urlDraft}
            onChange={(event) => setUrlDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") connectFromDraft();
            }}
          />
          <button onClick={connectFromDraft}>
            {connected ? "Reconnect" : "Connect"}
          </button>
        </div>
      </section>

      <section className="telemetry-ribbon">
        <article>
          <span>Grasp type</span>
          <strong>{ros.debug?.grasp_type ?? "—"}</strong>
        </article>
        <article>
          <span>Pose type</span>
          <strong>{ros.debug?.pose_type ?? "—"}</strong>
        </article>
        <article>
          <span>Controller phase</span>
          <strong>{ros.debug?.controller_phase || "—"}</strong>
        </article>
        <article className="coordinate-metric">
          <span>Geometric centroid Cg</span>
          <strong>{pointText(ros.debug?.geometric_centroid)}</strong>
        </article>
        <article className="coordinate-metric">
          <span>Virtual centroid Cv</span>
          <strong>{pointText(ros.debug?.virtual_centroid)}</strong>
        </article>
      </section>

      <section className="workspace">
        <div className="visual-column">
          <HandScene3D
            jointState={ros.jointState}
            debug={ros.debug}
            handToWorldRotation={ros.handToWorldRotation}
            orientationFromTopic={ros.lastRotationAt !== null}
          />
          <ForceHistoryPanel
            debug={ros.debug}
            live={debugLive}
            handToWorldRotation={ros.handToWorldRotation}
            orientationFromTopic={ros.lastRotationAt !== null}
          />
          <DebugReadout debug={ros.debug} />
        </div>
        <ControlPanel
          connected={connected}
          ready={controlsReady}
          debug={ros.debug}
          onGrasp={ros.setGraspType}
          onPose={ros.setPoseType}
          onAlpha={ros.setAlpha1}
          onTeaching={ros.setTeachingMode}
          onRotationMatrix={ros.setRotationMatrix}
          onNotice={pushNotice}
        />
      </section>

      <JointTable jointState={ros.jointState} debug={ros.debug} />

      <footer className={`footer-message footer-${notice.tone}`}>
        <div>
          <span className="footer-dot" />
          <strong>{notice.message}</strong>
        </div>
        <span>Calculated force · not a measured contact force</span>
      </footer>
    </main>
  );
}
