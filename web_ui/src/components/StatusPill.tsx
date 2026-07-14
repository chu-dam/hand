interface StatusPillProps {
  label: string;
  value: string;
  tone?: "ok" | "wait" | "danger" | "neutral";
}

export function StatusPill({
  label,
  value,
  tone = "neutral",
}: StatusPillProps) {
  return (
    <div className={`status-pill status-${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

