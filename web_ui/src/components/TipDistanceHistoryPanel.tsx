import { useEffect, useMemo, useRef, useState } from "react";

import type { TipAreaSample } from "../ros/types";

interface Props {
  sample: TipAreaSample | null;
}

interface Point {
  elapsed: number;
  areaMm2: number;
}

const AREA_REFERENCE_MM2 = 2_550;

export function TipAreaHistoryPanel({ sample }: Props) {
  const [history, setHistory] = useState<Point[]>([]);
  const originRef = useRef<number | null>(null);

  useEffect(() => {
    if (sample === null) return;
    if (originRef.current === null) originRef.current = sample.stampSec;
    setHistory((current) => {
      return [...current, {
        elapsed: sample.stampSec - originRef.current!,
        areaMm2: sample.areaM2 * 1_000_000,
      }].slice(-500);
    });
  }, [sample]);

  const width = 1080;
  const height = 230;
  const margin = { top: 18, right: 24, bottom: 38, left: 58 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const latest = history.at(-1)?.areaMm2;
  const { path, xMax, yMin, yMax } = useMemo(() => {
    const values = history.map((point) => point.areaMm2);
    const minimum = Math.min(AREA_REFERENCE_MM2, values.length ? Math.min(...values) : 2_600);
    const maximum = Math.max(AREA_REFERENCE_MM2, values.length ? Math.max(...values) : 2_700);
    const padding = Math.max(1, (maximum - minimum) * 0.15);
    const nextYMin = Math.floor(minimum - padding);
    const nextYMax = Math.ceil(maximum + padding);
    const nextXMax = Math.max(1, history.at(-1)?.elapsed ?? 0);
    const nextPath = history.map((point, index) => {
      const x = margin.left + point.elapsed / nextXMax * plotWidth;
      const y = margin.top + (nextYMax - point.areaMm2) / (nextYMax - nextYMin) * plotHeight;
      return `${index === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
    }).join("");
    return { path: nextPath, xMax: nextXMax, yMin: nextYMin, yMax: nextYMax };
  }, [history, margin.left, margin.top, plotHeight, plotWidth]);

  return (
    <section className="panel tip-distance-panel" aria-labelledby="tip-distance-title">
      <div className="panel-head force-history-head">
        <div>
          <p className="section-kicker">BASIC TIP FK HISTORY</p>
          <h2 id="tip-distance-title">Thumb–Index–Ring Triangle Area</h2>
          <p>link_base frame · one sample after thumb regrasp, before pinky release</p>
        </div>
        <div className="force-history-tools">
          <span className="force-history-status live">
            {latest === undefined ? "WAIT DATA" : `${latest.toFixed(3)} mm²`}
          </span>
          <button type="button" className="history-reset-button" onClick={() => { originRef.current = null; setHistory([]); }}>Reset</button>
        </div>
      </div>
      <svg className="tip-distance-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Thumb index ring fingertip triangle area over time">
        <rect className="force-chart-plot" x={margin.left} y={margin.top} width={plotWidth} height={plotHeight} />
        {[0, 0.5, 1].map((ratio) => {
          const x = margin.left + ratio * plotWidth;
          return <g key={`x-${ratio}`}><line className="force-chart-grid-line" x1={x} x2={x} y1={margin.top} y2={margin.top + plotHeight} /><text className="force-chart-tick" x={x} y={height - 18} textAnchor="middle">{(xMax * ratio).toFixed(1)}</text></g>;
        })}
        {Array.from({ length: yMax - yMin + 1 }, (_, index) => yMin + index).map((value) => {
          const y = margin.top + (yMax - value) / (yMax - yMin) * plotHeight;
          const labelEvery = Math.max(1, Math.ceil((yMax - yMin) / 6));
          return <g key={`y-${value}`}><line className="force-chart-grid-line" x1={margin.left} x2={margin.left + plotWidth} y1={y} y2={y} />{(value - yMin) % labelEvery === 0 && <text className="force-chart-tick" x={margin.left - 8} y={y + 3} textAnchor="end">{value}</text>}</g>;
        })}
        <line
          className="tip-area-reference"
          x1={margin.left}
          x2={margin.left + plotWidth}
          y1={margin.top + (yMax - AREA_REFERENCE_MM2) / (yMax - yMin) * plotHeight}
          y2={margin.top + (yMax - AREA_REFERENCE_MM2) / (yMax - yMin) * plotHeight}
        />
        <text
          className="tip-area-reference-label"
          x={margin.left + plotWidth - 4}
          y={margin.top + (yMax - AREA_REFERENCE_MM2) / (yMax - yMin) * plotHeight - 4}
          textAnchor="end"
        >
          REF 2550 mm²
        </text>
        <path className="tip-distance-line" d={path} />
        {history.map((point, index) => {
          const x = margin.left + point.elapsed / xMax * plotWidth;
          const y = margin.top + (yMax - point.areaMm2) / (yMax - yMin) * plotHeight;
          return <g key={`${point.elapsed}-${index}`}><circle className="tip-distance-point" cx={x} cy={y} r="3.5" /><text className="tip-distance-value" x={x} y={y - 7} textAnchor="middle">{point.areaMm2.toFixed(3)}</text></g>;
        })}
        <text className="force-chart-axis-label" x={margin.left + plotWidth / 2} y={height - 3} textAnchor="middle">Time [s]</text>
        <text className="force-chart-axis-label" x={-(margin.top + plotHeight / 2)} y="12" textAnchor="middle" transform="rotate(-90)">Area [mm²]</text>
        {history.length === 0 && <text className="force-chart-empty" x={margin.left + plotWidth / 2} y={margin.top + plotHeight / 2} textAnchor="middle">Waiting for first completed thumb regrasp</text>}
      </svg>
    </section>
  );
}
