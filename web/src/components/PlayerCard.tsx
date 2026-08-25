import { Link } from "react-router-dom";
import { useState } from "react";
import type { Player } from "../lib/types";
import { fdrFromFixture, money, num, statusDot } from "../lib/format";

type Props = {
  player: Player;
  captain?: "C" | "VC" | null;
  bench?: boolean;
  compact?: boolean;
  pointsKey?: "xp_gw" | "event_points";
};

export function PlayerCard({ player, captain, bench, compact, pointsKey = "xp_gw" }: Props) {
  const [imgFailed, setImgFailed] = useState(false);
  const fdr = fdrFromFixture(player.next_fixture) ?? Math.round(player.fdr_mean || 3);
  const shirt = !imgFailed && player.shirt_url;
  const pts = pointsKey === "event_points" ? player.event_points : player.xp_gw;

  return (
    <Link
      to={`/players/${player.id}`}
      className={`card relative flex w-[7.4rem] flex-col overflow-hidden rounded-lg no-underline transition hover:-translate-y-0.5 ${
        bench ? "opacity-80" : ""
      }`}
    >
      {captain && (
        <span className="absolute left-1.5 top-1.5 z-10 rounded-full bg-accent px-1.5 text-[10px] font-bold text-ink">
          {captain}
        </span>
      )}
      <div className="relative flex h-20 items-center justify-center overflow-hidden bg-[#0e1420]">
        {shirt ? (
          <img
            src={shirt}
            alt=""
            className="h-[4.25rem] w-auto object-contain"
            onError={() => setImgFailed(true)}
          />
        ) : (
          <div className="text-lg text-mute">{player.web_name.slice(0, 2)}</div>
        )}
      </div>
      <div className="space-y-1 px-2 py-1.5">
        <div className="flex items-center gap-1">
          <span className={`h-1.5 w-1.5 rounded-full ${statusDot(player.status)}`} />
          <p className="truncate text-[13px] font-semibold leading-tight">{player.web_name}</p>
        </div>
        <div className="flex items-center justify-between text-[10px] text-mute">
          <span>
            {player.position} · {player.team_short}
          </span>
          <span className={`rounded px-1 font-medium fdr-${Math.min(5, Math.max(1, fdr))}`}>{fdr}</span>
        </div>
        {!compact && (
          <div className="flex items-center justify-between text-[11px]">
            <span className="text-mute">{money(player.price)}</span>
            <span className="tabular font-medium text-accent">{num(pts, pointsKey === "event_points" ? 0 : 1)}</span>
          </div>
        )}
      </div>
    </Link>
  );
}
