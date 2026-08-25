import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../lib/api";
import { Pitch } from "../components/Pitch";
import { num } from "../lib/format";

export function Live() {
  const queryClient = useQueryClient();
  const { data, isPending, isError, error, dataUpdatedAt } = useQuery({
    queryKey: ["live"],
    queryFn: () => api.live(false),
    refetchInterval: 60_000,
  });
  const refresh = useMutation({
    mutationFn: () => api.live(true),
    onSuccess: (payload) => queryClient.setQueryData(["live"], payload),
  });
  if (isError) return <p className="text-rose-300">{(error as Error).message}</p>;
  if (isPending || !data) return <p className="text-mute">Loading Live GW…</p>;

  const byId = new Map(data.squad.map((p) => [p.id, p]));
  const official = data.official;
  const modelXi = data.squad
    .filter((p) => data.xi_ids.includes(p.id))
    .reduce((s, p) => s + (p.event_points || 0), 0);
  const pts = official?.points ?? modelXi;
  const benchPts = official?.points_on_bench;

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-lg font-semibold">GW{data.event} Live</h1>
        <p className="text-sm text-mute">
          {num(pts, 0)} Pts
          {benchPts != null ? ` · ${num(benchPts, 0)} Bench` : ""}
          {official?.hits ? ` · −${official.hits} Hits` : ""}
          {official?.chip ? ` · ${official.chip}` : ""}
          {" · "}
          {data.source === "entry" ? "Official Picks" : "Model XI"}
          {" · "}
          {new Date(dataUpdatedAt).toLocaleTimeString()}
        </p>
        <button
          type="button"
          className="mt-2 rounded-md border border-line px-2 py-1 text-xs"
          disabled={refresh.isPending}
          onClick={() => refresh.mutate()}
        >
          {refresh.isPending ? "Refreshing…" : "Refresh Live"}
        </button>
        {data.source === "model" && (
          <p className="mt-1 text-sm text-amber-200">Official Picks Unavailable — Showing Model XI.</p>
        )}
        {official?.auto_subs && official.auto_subs.length > 0 && (
          <p className="mt-1 text-sm text-mute">
            Auto-Subs:{" "}
            {official.auto_subs
              .map((s) => `${byId.get(s.out_id)?.web_name ?? s.out_id} → ${byId.get(s.in_id)?.web_name ?? s.in_id}`)
              .join(" · ")}
          </p>
        )}
      </div>
      <Pitch
        players={data.squad}
        xiIds={data.xi_ids}
        benchIds={data.bench_ids}
        captainId={data.captain_id}
        viceId={data.vice_id}
        pointsKey="event_points"
      />
      <section className="card overflow-x-auto rounded-xl p-4">
        <table className="w-full min-w-[560px] text-left text-sm">
          <thead className="text-xs text-mute">
            <tr>
              {["Player", "Min", "Pts", "BP", "BPS"].map((h) => (
                <th key={h} className="pb-2 pr-3 font-medium">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.squad.map((p) => (
              <tr key={p.id} className="border-t border-line">
                <td className="py-1.5 pr-3">
                  {p.web_name} <span className="text-mute">{p.position}</span>
                </td>
                <td className="tabular pr-3">{p.gw_minutes ?? p.minutes ?? "—"}</td>
                <td className="tabular pr-3">{p.event_points ?? 0}</td>
                <td className="tabular pr-3">{p.bonus ?? 0}</td>
                <td className="tabular pr-3">{p.bps ?? 0}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  );
}
