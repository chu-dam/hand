import { useEffect, useId, useMemo, useRef, useState } from "react";

import { rotateVectorToWorld } from "../ros/frames";
import {
  decodeFingerIds,
  type GraspDebugMessage,
  type Point3,
  type RotationMatrix3,
} from "../ros/types";

const EXPECTED_FINGER_COUNT = 5;
const MIN_SAMPLE_INTERVAL_MS = 40;
const MAX_CONTINUOUS_SAMPLE_GAP_MS = 500;
const MAX_HISTORY_SAMPLES = 6_000;
const MAX_RENDER_SAMPLES = 720;

const FINGER_COLORS = ["#c84b42", "#0b8f8f", "#5965bd", "#d3920b", "#7d54a5"];
const SUM_COLOR = "#172229";

const AXES = [
  { key: "x", label: "X", sumLabel: "ΣFworld,x" },
  { key: "y", label: "Y", sumLabel: "ΣFworld,y" },
  { key: "z", label: "Z", sumLabel: "ΣFworld,z" },
] as const;

type ForceAxis = (typeof AXES)[number]["key"];

interface ForceSnapshot {
  fingerIds: number[];
  fingers: Array<Point3 | null>;
  sum: Point3 | null;
}

interface ForceHistorySample {
  elapsed: number;
  fingers: Array<Point3 | null>;
  sum: Point3 | null;
  breakBefore: boolean;
}

interface ForceHistoryPanelProps {
  debug: GraspDebugMessage | null;
  live: boolean;
  handToWorldRotation: RotationMatrix3;
  orientationFromTopic: boolean;
}

interface ForceAxisChartProps {
  axis: ForceAxis;
  axisLabel: string;
  sumLabel: string;
  currentSum: number | null;
  samples: ForceHistorySample[];
  emptyLabel: string;
}

function finitePoint(point: Point3 | undefined): Point3 | null {
  if (
    !point
    || !Number.isFinite(point.x)
    || !Number.isFinite(point.y)
    || !Number.isFinite(point.z)
  ) {
    return null;
  }

  return { x: point.x, y: point.y, z: point.z };
}

function snapshotFromDebug(
  debug: GraspDebugMessage | null,
  handToWorldRotation: RotationMatrix3,
): ForceSnapshot | null {
  if (
    !debug
    || debug.header.frame_id !== "link_base"
    || !Array.isArray(debug.total_forces)
  ) {
    return null;
  }

  const decodedIds = decodeFingerIds(debug.finger_ids);
  const fingerIds = Array.from(
    { length: EXPECTED_FINGER_COUNT },
    (_, index) => decodedIds[index] ?? index + 1,
  );
  const fingers = Array.from(
    { length: EXPECTED_FINGER_COUNT },
    (_, index) => {
      const forceInHand = finitePoint(debug.total_forces[index]);
      return forceInHand === null
        ? null
        : rotateVectorToWorld(forceInHand, handToWorldRotation);
    },
  );

  const sum: Point3 = { x: 0, y: 0, z: 0 };
  for (const point of fingers) {
    // The debug contract requires exactly five finite vectors. An incomplete
    // packet is ignored instead of being mistaken for a zero-force sample.
    if (point === null) return null;
    sum.x += point.x;
    sum.y += point.y;
    sum.z += point.z;
  }

  return {
    fingerIds,
    fingers,
    sum,
  };
}

function sampleValue(
  sample: ForceHistorySample,
  axis: ForceAxis,
  seriesIndex: number,
): number | null {
  const point = seriesIndex < EXPECTED_FINGER_COUNT
    ? sample.fingers[seriesIndex]
    : sample.sum;
  return point?.[axis] ?? null;
}

function compactHistory(samples: ForceHistorySample[]): ForceHistorySample[] {
  if (samples.length <= MAX_HISTORY_SAMPLES) return samples;

  // Keep the full span since reset and all old per-series extrema while
  // reducing only redundant points. Recent samples remain at full rate.
  const olderEnd = Math.floor(MAX_HISTORY_SAMPLES / 2);
  const targetOlderSamples = Math.floor(MAX_HISTORY_SAMPLES / 4);
  const maximumExtremaPerBucket = AXES.length * (EXPECTED_FINGER_COUNT + 1) * 2 + 2;
  const bucketCount = Math.max(1, Math.floor(targetOlderSamples / maximumExtremaPerBucket));
  const bucketSize = Math.ceil(olderEnd / bucketCount);
  const selectedIndices = new Set<number>();

  for (let start = 0; start < olderEnd; start += bucketSize) {
    const end = Math.min(olderEnd - 1, start + bucketSize - 1);
    selectedIndices.add(start);
    selectedIndices.add(end);

    for (const { key: axis } of AXES) {
      for (let seriesIndex = 0; seriesIndex <= EXPECTED_FINGER_COUNT; seriesIndex += 1) {
        let minimumIndex = -1;
        let maximumIndex = -1;
        let minimum = Number.POSITIVE_INFINITY;
        let maximum = Number.NEGATIVE_INFINITY;

        for (let index = start; index <= end; index += 1) {
          const value = sampleValue(samples[index], axis, seriesIndex);
          if (value === null || !Number.isFinite(value)) continue;
          if (value < minimum) {
            minimum = value;
            minimumIndex = index;
          }
          if (value > maximum) {
            maximum = value;
            maximumIndex = index;
          }
        }

        if (minimumIndex >= 0) selectedIndices.add(minimumIndex);
        if (maximumIndex >= 0) selectedIndices.add(maximumIndex);
      }
    }

  }

  const sortedIndices = [...selectedIndices].sort((left, right) => left - right);
  let previousIndex = -1;
  const older = sortedIndices.map((index) => {
    let breakBefore = samples[index].breakBefore;
    for (let skipped = previousIndex + 1; skipped <= index; skipped += 1) {
      if (samples[skipped].breakBefore) {
        breakBefore = true;
        break;
      }
    }
    previousIndex = index;
    return breakBefore === samples[index].breakBefore
      ? samples[index]
      : { ...samples[index], breakBefore };
  });
  return [...older, ...samples.slice(olderEnd)];
}

interface ForceSeriesPoint {
  elapsed: number;
  value: number | null;
  breakBefore: boolean;
}

function seriesForRendering(
  samples: ForceHistorySample[],
  axis: ForceAxis,
  seriesIndex: number,
): ForceSeriesPoint[] {
  if (samples.length <= MAX_RENDER_SAMPLES) {
    return samples.map((sample) => ({
      elapsed: sample.elapsed,
      value: sampleValue(sample, axis, seriesIndex),
      breakBefore: sample.breakBefore,
    }));
  }

  // First/last plus min/max from every bucket preserves short force spikes
  // without sending thousands of SVG points to the browser each frame.
  const maximumPointsPerBucket = 4;
  const bucketCount = Math.max(1, Math.floor(MAX_RENDER_SAMPLES / maximumPointsPerBucket));
  const bucketSize = Math.ceil(samples.length / bucketCount);
  const selectedIndices = new Set<number>();

  for (let start = 0; start < samples.length; start += bucketSize) {
    const end = Math.min(samples.length - 1, start + bucketSize - 1);
    let minimumIndex = -1;
    let maximumIndex = -1;
    let minimum = Number.POSITIVE_INFINITY;
    let maximum = Number.NEGATIVE_INFINITY;
    selectedIndices.add(start);
    selectedIndices.add(end);

    for (let index = start; index <= end; index += 1) {
      const value = sampleValue(samples[index], axis, seriesIndex);
      if (value !== null && Number.isFinite(value)) {
        if (value < minimum) {
          minimum = value;
          minimumIndex = index;
        }
        if (value > maximum) {
          maximum = value;
          maximumIndex = index;
        }
      }
    }

    if (minimumIndex >= 0) selectedIndices.add(minimumIndex);
    if (maximumIndex >= 0) selectedIndices.add(maximumIndex);
  }

  const indices = [...selectedIndices].sort((left, right) => left - right);
  let previousIndex = -1;
  return indices.map((index) => {
    let breakBefore = samples[index].breakBefore;
    for (let skipped = previousIndex + 1; skipped <= index; skipped += 1) {
      if (
        samples[skipped].breakBefore
        || sampleValue(samples[skipped], axis, seriesIndex) === null
      ) {
        breakBefore = true;
        break;
      }
    }
    previousIndex = index;
    return {
      elapsed: samples[index].elapsed,
      value: sampleValue(samples[index], axis, seriesIndex),
      breakBefore,
    };
  });
}

function niceCeiling(value: number): number {
  if (!Number.isFinite(value) || value <= 0) return 1;
  const exponent = Math.floor(Math.log10(value));
  const magnitude = 10 ** exponent;
  const fraction = value / magnitude;
  const niceFraction = fraction <= 1 ? 1 : fraction <= 2 ? 2 : fraction <= 5 ? 5 : 10;
  return niceFraction * magnitude;
}

function currentValue(value: number | null): string {
  if (value === null || !Number.isFinite(value)) return "—";
  const normalized = Math.abs(value) < 0.0005 ? 0 : value;
  return `${normalized > 0 ? "+" : ""}${normalized.toFixed(3)}`;
}

function tickValue(value: number, limit: number): string {
  const digits = limit < 1 ? 2 : limit < 10 ? 1 : 0;
  const normalized = Math.abs(value) < 1e-10 ? 0 : value;
  return normalized.toFixed(digits);
}

function ForceAxisChart({
  axis,
  axisLabel,
  sumLabel,
  currentSum,
  samples,
  emptyLabel,
}: ForceAxisChartProps) {
  const rawClipId = useId();
  const clipId = `force-chart-${rawClipId.replace(/:/g, "")}`;

  const width = 360;
  const height = 220;
  const margin = { top: 14, right: 12, bottom: 34, left: 48 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;

  const { paths, xMax, yLimit } = useMemo(() => {
    const latestElapsed = samples.at(-1)?.elapsed ?? 0;
    const nextXMax = niceCeiling(Math.max(5, latestElapsed));
    let maximumMagnitude = 0;

    for (const sample of samples) {
      for (const point of sample.fingers) {
        if (point !== null) maximumMagnitude = Math.max(maximumMagnitude, Math.abs(point[axis]));
      }
      if (sample.sum !== null) maximumMagnitude = Math.max(maximumMagnitude, Math.abs(sample.sum[axis]));
    }

    const nextYLimit = niceCeiling(Math.max(1, maximumMagnitude * 1.1));
    const nextPaths = Array.from({ length: EXPECTED_FINGER_COUNT + 1 }, (_, seriesIndex) => {
      let path = "";
      let drawing = false;

      for (const point of seriesForRendering(samples, axis, seriesIndex)) {
        const { value } = point;
        if (value === null || !Number.isFinite(value)) {
          drawing = false;
          continue;
        }

        const x = margin.left + (point.elapsed / nextXMax) * plotWidth;
        const y = margin.top + ((nextYLimit - value) / (nextYLimit * 2)) * plotHeight;
        const command = !drawing || point.breakBefore ? "M" : "L";
        path += `${command}${x.toFixed(2)},${y.toFixed(2)}`;
        drawing = true;
      }

      return path;
    });

    return { paths: nextPaths, xMax: nextXMax, yLimit: nextYLimit };
  }, [axis, margin.left, margin.top, plotHeight, plotWidth, samples]);

  const xTicks = Array.from({ length: 6 }, (_, index) => (xMax * index) / 5);
  const yTicks = Array.from({ length: 5 }, (_, index) => yLimit - (yLimit * 2 * index) / 4);

  return (
    <article className="force-axis-card">
      <div className="force-axis-card-head">
        <span className={`force-axis-token axis-${axis}`}>{axisLabel}</span>
        <div>
          <strong>World {axisLabel}-axis Force</strong>
          <small>{sumLabel} <b>{currentValue(currentSum)} N</b></small>
        </div>
      </div>

      <svg
        className="force-chart"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`World ${axisLabel}-axis calculated force history by finger`}
      >
        <title>World {axisLabel}-axis calculated force history</title>
        <desc>World-frame calculated forces for F1 through F5 and their signed axis sum over time.</desc>
        <defs>
          <clipPath id={clipId}>
            <rect x={margin.left} y={margin.top} width={plotWidth} height={plotHeight} />
          </clipPath>
        </defs>

        <rect
          className="force-chart-plot"
          x={margin.left}
          y={margin.top}
          width={plotWidth}
          height={plotHeight}
        />

        {xTicks.map((tick) => {
          const x = margin.left + (tick / xMax) * plotWidth;
          return (
            <g key={`x-${tick}`}>
              <line className="force-chart-grid-line" x1={x} x2={x} y1={margin.top} y2={margin.top + plotHeight} />
              <text className="force-chart-tick" x={x} y={height - 19} textAnchor="middle">
                {tickValue(tick, xMax)}
              </text>
            </g>
          );
        })}

        {yTicks.map((tick) => {
          const y = margin.top + ((yLimit - tick) / (yLimit * 2)) * plotHeight;
          return (
            <g key={`y-${tick}`}>
              <line
                className={tick === 0 ? "force-chart-zero" : "force-chart-grid-line"}
                x1={margin.left}
                x2={margin.left + plotWidth}
                y1={y}
                y2={y}
              />
              <text className="force-chart-tick" x={margin.left - 7} y={y + 3} textAnchor="end">
                {tickValue(tick, yLimit)}
              </text>
            </g>
          );
        })}

        <g clipPath={`url(#${clipId})`}>
          {paths.slice(0, EXPECTED_FINGER_COUNT).map((path, index) => (
            <path
              className="force-chart-series"
              d={path}
              key={`finger-${index}`}
              stroke={FINGER_COLORS[index]}
            />
          ))}
          <path className="force-chart-series force-chart-sum" d={paths[EXPECTED_FINGER_COUNT]} stroke={SUM_COLOR} />
        </g>

        <text className="force-chart-axis-label" x={margin.left + plotWidth / 2} y={height - 3} textAnchor="middle">
          Time [s]
        </text>
        <text
          className="force-chart-axis-label"
          x={-(margin.top + plotHeight / 2)}
          y={11}
          textAnchor="middle"
          transform="rotate(-90)"
        >
          Force [N]
        </text>

        {samples.length === 0 && (
          <text
            className="force-chart-empty"
            x={margin.left + plotWidth / 2}
            y={margin.top + plotHeight / 2}
            textAnchor="middle"
          >
            {emptyLabel}
          </text>
        )}
      </svg>
    </article>
  );
}

export function ForceHistoryPanel({
  debug,
  live,
  handToWorldRotation,
  orientationFromTopic,
}: ForceHistoryPanelProps) {
  const [history, setHistory] = useState<ForceHistorySample[]>([]);
  const [sampleCount, setSampleCount] = useState(0);
  const originRef = useRef<number | null>(null);
  const lastAcceptedAtRef = useRef(Number.NEGATIVE_INFINITY);
  const lastDebugRef = useRef<GraspDebugMessage | null>(null);
  const breakBeforeRef = useRef(true);

  const snapshot = useMemo(
    () => snapshotFromDebug(debug, handToWorldRotation),
    [debug, handToWorldRotation],
  );
  const fingerIds = snapshot?.fingerIds ?? [1, 2, 3, 4, 5];
  const elapsed = history.at(-1)?.elapsed ?? 0;

  useEffect(() => {
    if (
      debug === null
      || debug.header.frame_id !== "link_base"
    ) {
      lastDebugRef.current = debug;
      breakBeforeRef.current = true;
      return;
    }
    if (lastDebugRef.current === debug) return;
    lastDebugRef.current = debug;

    const nextSnapshot = snapshotFromDebug(debug, handToWorldRotation);
    if (nextSnapshot === null) return;

    const now = performance.now();
    const sampleGap = now - lastAcceptedAtRef.current;
    if (sampleGap < MIN_SAMPLE_INTERVAL_MS) return;
    const breakBefore = breakBeforeRef.current || sampleGap > MAX_CONTINUOUS_SAMPLE_GAP_MS;
    lastAcceptedAtRef.current = now;

    if (originRef.current === null) originRef.current = now;
    const sample: ForceHistorySample = {
      elapsed: Math.max(0, (now - originRef.current) / 1_000),
      fingers: nextSnapshot.fingers,
      sum: nextSnapshot.sum,
      breakBefore,
    };
    breakBeforeRef.current = false;

    setHistory((current) => compactHistory([...current, sample]));
    setSampleCount((current) => current + 1);
  }, [debug, handToWorldRotation]);

  const resetTime = () => {
    originRef.current = performance.now();
    lastAcceptedAtRef.current = Number.NEGATIVE_INFINITY;
    breakBeforeRef.current = true;
    setHistory([]);
    setSampleCount(0);
  };

  const sourceFrameReady = debug === null || debug.header.frame_id === "link_base";
  const historyState = !sourceFrameReady
    ? "WAIT FRAME"
    : live && history.length > 0
      ? `LIVE · ${elapsed.toFixed(1)} s`
      : history.length > 0
        ? `PAUSED · ${elapsed.toFixed(1)} s`
        : "WAIT DATA";
  const emptyLabel = sourceFrameReady
    ? "Waiting for GraspDebug"
    : `Unsupported source frame: ${debug?.header.frame_id || "—"}`;

  return (
    <section className="panel force-history-panel" aria-labelledby="force-history-title">
      <div className="panel-head force-history-head">
        <div>
          <p className="section-kicker">FORCE HISTORY</p>
          <h2 id="force-history-title">World X/Y/Z Force History</h2>
          <p>
            {debug?.header.frame_id || "link_base"} → world · R_hand_to_world ({orientationFromTopic ? "topic" : "default identity"}) × displayed fingertip force · elapsed time since reset
          </p>
        </div>
        <div className="force-history-tools">
          <span className={`force-history-status ${sourceFrameReady && live ? "live" : "waiting"}`}>
            {historyState}
          </span>
          <button type="button" className="history-reset-button" onClick={resetTime}>
            Reset
          </button>
        </div>
      </div>

      <div className="force-chart-grid">
        {AXES.map(({ key, label, sumLabel }) => (
          <ForceAxisChart
            axis={key}
            axisLabel={label}
            currentSum={snapshot?.sum?.[key] ?? null}
            emptyLabel={emptyLabel}
            key={key}
            samples={history}
            sumLabel={sumLabel}
          />
        ))}
      </div>

      <div className="force-history-footer">
        <div className="force-history-legend" aria-label="Chart legend">
          {fingerIds.map((id, index) => (
            <span key={`${id}-${index}`}>
              <i style={{ backgroundColor: FINGER_COLORS[index] }} />F{id}
            </span>
          ))}
          <span><i className="sum-line" />Sum</span>
        </div>
        <span className="force-history-note">
          world frame · 0 N reference · Cartesian command forces (not sensor measurements) · {sampleCount.toLocaleString()} samples
        </span>
      </div>
    </section>
  );
}
