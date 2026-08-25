import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";

export function Fixtures() {
  const { data, isPending } = useQuery({ queryKey: ["fixtures"], queryFn: () => api.fixtures() });
  if (isPending || !data) return <p className="text-mute">Loading Fixtures…</p>;

  const byTeam = new Map<number, Map<number, (typeof data.cells)[number]>>();
  for (const cell of data.cells) {
    if (!byTeam.has(cell.team_id)) byTeam.set(cell.team_id, new Map());
    byTeam.get(cell.team_id)!.set(cell.event, cell);
  }

  return (
    <div className="card overflow-x-auto rounded-xl p-4">
      <h1 className="mb-4 text-lg font-semibold">FDR Heatmap</h1>
      <table className="w-full min-w-[720px] text-left text-sm">
        <thead>
          <tr className="text-xs text-mute">
            <th className="pb-2 pr-3">Team</th>
            {data.events.map((gw) => (
              <th key={gw} className="pb-2 pr-2 text-center">
                GW{gw}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.teams.map((team) => (
            <tr key={team.team_id} className="border-t border-line">
              <td className="py-1.5 pr-3">
                <span className="inline-flex items-center gap-2">
                  {team.badge_url && <img src={team.badge_url} alt="" className="h-5 w-5" />}
                  {team.team_short}
                </span>
              </td>
              {data.events.map((gw) => {
                const cell = byTeam.get(team.team_id)?.get(gw);
                if (!cell) {
                  return (
                    <td key={gw} className="pr-2 text-center">
                      <span className="inline-block w-full rounded bg-white/5 px-1 py-1 text-[11px] text-mute">—</span>
                    </td>
                  );
                }
                return (
                  <td key={gw} className="pr-2 text-center">
                    <span className={`inline-block w-full rounded px-1 py-1 text-[11px] fdr-${cell.fdr}`}>
                      {cell.home ? "H" : "A"} {cell.opponent}
                    </span>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
