import type { Player } from "../lib/types";
import { PlayerCard } from "./PlayerCard";

const ROWS: Array<Player["position"]> = ["GKP", "DEF", "MID", "FWD"];

type Props = {
  players: Player[];
  xiIds: number[];
  benchIds: number[];
  captainId?: number | null;
  viceId?: number | null;
  pointsKey?: "xp_gw" | "event_points";
};

export function Pitch({ players, xiIds, benchIds, captainId, viceId, pointsKey }: Props) {
  const byId = new Map(players.map((p) => [p.id, p]));
  const xi = xiIds.map((id) => byId.get(id)).filter(Boolean) as Player[];
  const bench = benchIds.map((id) => byId.get(id)).filter(Boolean) as Player[];

  const role = (id: number) => (id === captainId ? "C" : id === viceId ? "VC" : null);

  return (
    <div className="pitch-bg overflow-hidden rounded-2xl border border-white/10 p-4 md:p-6">
      <div className="space-y-5">
        {ROWS.map((pos) => {
          const row = xi.filter((p) => p.position === pos);
          return (
            <div key={pos} className="flex flex-wrap justify-center gap-3">
              {row.map((p) => (
                <PlayerCard key={p.id} player={p} captain={role(p.id)} pointsKey={pointsKey} />
              ))}
            </div>
          );
        })}
      </div>
      {bench.length > 0 && (
        <div className="mt-6 border-t border-white/15 pt-4">
          <p className="mb-2 text-center text-[11px] uppercase tracking-[0.18em] text-white/60">Bench</p>
          <div className="flex flex-wrap justify-center gap-3">
            {bench.map((p, i) => (
              <div key={p.id} className="relative">
                <span className="absolute -left-1 -top-1 z-10 flex h-4 w-4 items-center justify-center rounded-full bg-black/50 text-[10px] text-white">
                  {i + 1}
                </span>
                <PlayerCard player={p} bench pointsKey={pointsKey} />
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
