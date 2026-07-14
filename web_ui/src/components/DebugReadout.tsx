import { decodeFingerIds, vectorMagnitude, type GraspDebugMessage } from "../ros/types";

interface DebugReadoutProps {
  debug: GraspDebugMessage | null;
}

function fmt(value: number | undefined, digits = 3) {
  return value === undefined || !Number.isFinite(value) ? "—" : value.toFixed(digits);
}

export function DebugReadout({ debug }: DebugReadoutProps) {
  const ids = debug ? decodeFingerIds(debug.finger_ids) : [1, 2, 3, 4, 5];

  return (
    <section className="panel debug-readout">
      <div className="panel-head compact-head">
        <div>
          <p className="section-kicker">LIVE VALUES</p>
          <h2>Finger force summary</h2>
        </div>
        <span className="frame-badge">{debug?.header.frame_id || "frame —"}</span>
      </div>
      <div className="finger-summary-grid">
        {ids.map((id, index) => (
          <article className="finger-summary" key={id}>
            <div className="finger-summary-title">
              <span>F{id}</span>
              <strong>{fmt(vectorMagnitude(debug?.total_forces[index]), 2)} N</strong>
            </div>
            <dl>
              <div><dt>alpha</dt><dd>{fmt(debug?.alpha[index])}</dd></div>
              <div><dt>grasp</dt><dd>{fmt(vectorMagnitude(debug?.grasp_forces[index]))}</dd></div>
              <div><dt>rotation</dt><dd>{fmt(vectorMagnitude(debug?.rotation_forces[index]))}</dd></div>
              <div><dt>center hold</dt><dd>{fmt(vectorMagnitude(debug?.center_hold_forces[index]))}</dd></div>
              <div><dt>collision</dt><dd>{fmt(vectorMagnitude(debug?.collision_forces[index]))}</dd></div>
            </dl>
          </article>
        ))}
      </div>
    </section>
  );
}
