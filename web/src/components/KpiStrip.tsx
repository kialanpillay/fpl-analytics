import type { Analysis } from "../lib/types";
import { money, num } from "../lib/format";

export function KpiStrip({ analysis }: { analysis: Analysis }) {
  const ev = analysis.squad_eval;
  const xiXp = analysis.squad
    .filter((p) => analysis.xi_ids.includes(p.id))
    .reduce((s, p) => s + (p.xp_gw || 0), 0);
  const items = [
    ["Squad", money(ev.cost)],
    ["Bank", money(analysis.meta.bank)],
    ["XI xPts", num(xiXp, 1)],
    ["Horizon", num(ev.xp_horizon, 1)],
    ["PPP", num(ev.ppp, 2)],
    ["Cons", num(ev.consistency, 2)],
    ["Dead", String(ev.dead_slots.length)],
    ["FTs", String(analysis.meta.free_transfers)],
  ];
  return (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-4 lg:grid-cols-8">
      {items.map(([label, value]) => (
        <div key={label} className="card rounded-lg px-3 py-2">
          <p className="text-[11px] font-medium text-mute">{label}</p>
          <p className="tabular text-lg font-semibold">{value}</p>
        </div>
      ))}
    </div>
  );
}
