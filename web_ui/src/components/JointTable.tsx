import type { GraspDebugMessage, JointStateMessage } from "../ros/types";

const JOINT_NAMES = Array.from(
  { length: 20 },
  (_, index) => `joint_${Math.floor(index / 4) + 1}_${(index % 4) + 1}`,
);

const FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"];

interface JointTableProps {
  jointState: JointStateMessage | null;
  debug: GraspDebugMessage | null;
}

function fmt(value: number | undefined, digits = 4) {
  return value === undefined || !Number.isFinite(value) ? "—" : value.toFixed(digits);
}

export function JointTable({ jointState, debug }: JointTableProps) {
  const stateIndex = new Map(
    (jointState?.name ?? []).map((name, index) => [name, index]),
  );

  return (
    <section className="panel joint-panel">
      <div className="panel-head compact-head">
        <div>
          <p className="section-kicker">20 DOF STREAM</p>
          <h2>Joint state &amp; control output</h2>
        </div>
        <span className="table-note">name-based mapping</span>
      </div>
      <div className="joint-table-wrap">
        <table className="joint-table">
          <thead>
            <tr>
              <th>Finger</th>
              <th>Joint</th>
              <th>Deg</th>
              <th>Rad</th>
              <th>Velocity</th>
              <th>Controller τ</th>
              <th>Commanded</th>
              <th className="current-heading">FEEDBACK I (mA)</th>
            </tr>
          </thead>
          <tbody>
            {JOINT_NAMES.map((name, fixedIndex) => {
              const messageIndex = stateIndex.get(name);
              const position = messageIndex === undefined ? undefined : jointState?.position[messageIndex];
              const velocity = messageIndex === undefined ? undefined : jointState?.velocity[messageIndex];
              const current = messageIndex === undefined ? undefined : jointState?.effort[messageIndex];
              return (
                <tr key={name}>
                  <td><span className={`finger-tag tag-${Math.floor(fixedIndex / 4) + 1}`}>{FINGER_NAMES[Math.floor(fixedIndex / 4)]}</span></td>
                  <td className="joint-name">{name}</td>
                  <td>{fmt(position === undefined ? undefined : position * 180 / Math.PI, 2)}</td>
                  <td>{fmt(position)}</td>
                  <td>{fmt(velocity)}</td>
                  <td className="torque-value">{fmt(debug?.controller_torques[fixedIndex])}</td>
                  <td>{fmt(debug?.commanded_efforts[fixedIndex])}</td>
                  <td className="current-value">{fmt(current, 1)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}
