import type { Note, Strategy } from "../lib/types";
import { toneClass } from "../lib/format";

export function Notes({ notes, strategies, warnings }: { notes: Note[]; strategies: Strategy[]; warnings: string[] }) {
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <section className="card rounded-xl p-4">
        <h2 className="mb-3 text-sm font-semibold text-mute">Squad Notes</h2>
        <ul className="space-y-2 text-sm">
          {warnings.map((w) => (
            <li key={w} className="text-amber-300">
              {w}
            </li>
          ))}
          {notes.map((n) => (
            <li key={n.name}>
              <span className={`font-medium ${toneClass(n.tone)}`}>[{n.tone[0].toUpperCase() + n.tone.slice(1)}] {n.name}</span>
              <span className="text-slate-300"> — {n.note}</span>
            </li>
          ))}
        </ul>
      </section>
      <section className="card rounded-xl p-4">
        <h2 className="mb-3 text-sm font-semibold text-mute">Season Strategies</h2>
        <ul className="space-y-3 text-sm text-slate-300">
          {strategies.map((s) => (
            <li key={s.id}>
              <p className="font-medium text-white">{s.title}</p>
              <p className="text-mute">{s.detail}</p>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
