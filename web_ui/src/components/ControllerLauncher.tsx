import { useCallback, useEffect, useState } from "react";

import type { HandSide } from "../ros/types";

interface HandStatus {
  running: boolean;
  pid: number | null;
}

type ControllerStatus = Record<HandSide, HandStatus>;

interface ControllerLauncherProps {
  selectedHand: HandSide;
  onSelectHand: (side: HandSide) => void;
  onNotice: (message: string, tone?: "ok" | "warning" | "error") => void;
}

const EMPTY_STATUS: ControllerStatus = {
  left: { running: false, pid: null },
  right: { running: false, pid: null },
};

export function ControllerLauncher({
  selectedHand,
  onSelectHand,
  onNotice,
}: ControllerLauncherProps) {
  const [status, setStatus] = useState(EMPTY_STATUS);
  const [busy, setBusy] = useState<HandSide | null>(null);

  const refresh = useCallback(async () => {
    try {
      const response = await fetch("/api/controllers");
      if (!response.ok) throw new Error("controller API error");
      setStatus(await response.json() as ControllerStatus);
    } catch {
      setStatus(EMPTY_STATUS);
    }
  }, []);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(), 1_000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  const request = async (side: HandSide, action: "start" | "stop") => {
    setBusy(side);
    try {
      const response = await fetch(`/api/controllers/${side}/${action}`, { method: "POST" });
      const body = await response.json() as ControllerStatus & { error?: string };
      if (!response.ok) throw new Error(body.error || "controller API error");
      setStatus(body);
      if (action === "start") onSelectHand(side);
      onNotice(
        `${side === "left" ? "왼손" : "오른손"} 컨트롤러 ${action === "start" ? "활성화" : "종료"} 요청을 완료했습니다.`,
        action === "start" ? "ok" : "warning",
      );
    } catch (error) {
      onNotice(error instanceof Error ? error.message : "컨트롤러 요청 실패", "error");
    } finally {
      setBusy(null);
      void refresh();
    }
  };

  const handleAction = (side: HandSide, running: boolean) => {
    if (
      !running
      && !window.confirm(
        `${side === "left" ? "왼손" : "오른손"}에 즉시 토크가 적용되고 Normal pose로 움직입니다. 주변이 안전합니까?`,
      )
    ) return;
    void request(side, running ? "stop" : "start");
  };

  return (
    <section className="controller-launcher">
      {(["left", "right"] as HandSide[]).map((side) => {
        const label = side === "left" ? "LEFT HAND" : "RIGHT HAND";
        const running = status[side].running;
        return (
          <article key={side} className={`${selectedHand === side ? "selected" : ""} ${running ? "running" : ""}`}>
            <button className="hand-selector" onClick={() => onSelectHand(side)}>
              <span>{label}</span>
              <strong>{running ? "RUNNING" : selectedHand === side ? "SELECTED" : "STANDBY"}</strong>
            </button>
            <button
              className={running ? "controller-stop" : "controller-start"}
              disabled={busy !== null}
              onClick={() => handleAction(side, running)}
            >
              {busy === side ? "WAIT" : running ? "STOP" : "ACTIVATE"}
            </button>
          </article>
        );
      })}
    </section>
  );
}
