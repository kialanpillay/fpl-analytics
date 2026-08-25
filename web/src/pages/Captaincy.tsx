import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import { num } from "../lib/format";
import { PlayerCard } from "../components/PlayerCard";

export function Captaincy() {
  const { data, isPending } = useQuery({ queryKey: ["captaincy"], queryFn: api.captaincy });
  if (isPending || !data) return <p className="text-mute">Ranking Captains…</p>;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap gap-4">
        {data.recommended && (
          <div>
            <p className="mb-2 text-xs font-medium text-mute">Recommended C</p>
            <PlayerCard player={data.recommended} captain="C" />
          </div>
        )}
        {data.vice && (
          <div>
            <p className="mb-2 text-xs font-medium text-mute">Vice</p>
            <PlayerCard player={data.vice} captain="VC" />
          </div>
        )}
      </div>
      <section className="card overflow-x-auto rounded-xl p-4">
        <h1 className="mb-3 text-lg font-semibold">XI Ranked By xPts × Minutes</h1>
        <table className="w-full min-w-[640px] text-left text-sm">
          <thead className="text-xs text-mute">
            <tr>
              {["#", "Player", "Pos", "xPts", "Mins", "Score", "C EV", "Default"].map((h) => (
                <th key={h} className="pb-2 pr-3 font-medium">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.options.map((p) => (
              <tr key={p.id} className="border-t border-line">
                <td className="py-2 pr-3">{p.role}</td>
                <td className="pr-3">{p.web_name}</td>
                <td className="pr-3 text-mute">
                  {p.position} {p.team_short}
                </td>
                <td className="tabular pr-3">{num(p.xp_gw)}</td>
                <td className="tabular pr-3">{num(p.minutes_prob)}</td>
                <td className="tabular pr-3 text-accent">{num(p.captain_score)}</td>
                <td className="tabular pr-3">{num(p.captain_ev)}</td>
                <td>{p.season_default ? "Haaland" : ""}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  );
}
