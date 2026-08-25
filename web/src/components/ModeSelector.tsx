import { OBJECTIVE_MODES, type ObjectiveMode } from "../lib/modes";

export function ModeSelector({
  value,
  onChange,
}: {
  value: ObjectiveMode;
  onChange: (mode: ObjectiveMode) => void;
}) {
  return (
    <div className="flex flex-wrap gap-2">
      {OBJECTIVE_MODES.map(([id, label]) => (
        <button
          key={id}
          type="button"
          onClick={() => onChange(id)}
          className={`rounded-md px-3 py-1.5 text-sm ${value === id ? "bg-accent text-ink" : "bg-white/5 text-mute"}`}
        >
          {label}
        </button>
      ))}
    </div>
  );
}
