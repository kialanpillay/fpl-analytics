import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useParams } from "react-router-dom";
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../lib/api";
import { money, num, toneClass } from "../lib/format";

export function PlayerDetail() {
  const { id } = useParams();
  const playerId = Number(id);
  const [photoFailed, setPhotoFailed] = useState(false);
  const { data, isPending, error } = useQuery({
    queryKey: ["player", playerId],
    queryFn: () => api.player(playerId),
    enabled: Number.isFinite(playerId),
  });
  if (isPending) return <p className="text-mute">Loading Player…</p>;
  if (error || !data) return <p className="text-rose-300">{(error as Error)?.message || "Not Found"}</p>;

  const history = (data.history || []).map((row) => ({
    gw: Number(row.round ?? row.event ?? 0),
    points: Number(row.total_points ?? 0),
    minutes: Number(row.minutes ?? 0),
    xgi: Number(row.expected_goal_involvements ?? 0),
  }));
  const p = data.player;
  const portrait = !photoFailed && p.photo_url;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start gap-6">
        <div className="card h-52 w-40 overflow-hidden rounded-xl bg-[#0e1420]">
          {portrait ? (
            <img
              src={portrait}
              alt={p.web_name}
              className="h-full w-full object-cover object-top"
              onError={() => setPhotoFailed(true)}
            />
          ) : p.shirt_url ? (
            <img src={p.shirt_url} alt="" className="mx-auto mt-8 h-28 object-contain" />
          ) : (
            <div className="flex h-full items-center justify-center text-2xl text-mute">
              {p.web_name.slice(0, 2)}
            </div>
          )}
        </div>
        <div className="min-w-[16rem] flex-1">
          <h1 className="text-2xl font-semibold">
            {p.web_name}{" "}
            <span className="text-base font-normal text-mute">
              {p.position} {p.team_short} {money(p.price)}
            </span>
          </h1>
          {data.in_squad && <p className="text-xs font-medium tracking-wide text-accent">In Squad</p>}
          {data.note && (
            <p className={`mt-2 text-sm ${toneClass(data.note.tone)}`}>
              [{data.note.tone[0].toUpperCase() + data.note.tone.slice(1)}] {data.note.note}
            </p>
          )}
          {p.news && <p className="mt-2 text-sm text-amber-200">{p.news}</p>}
          <dl className="mt-4 grid grid-cols-2 gap-2 text-sm sm:grid-cols-4">
            {[
              ["xGW", num(p.xp_gw)],
              ["Horizon", num(p.xp_horizon)],
              ["PPP", num(p.ppp)],
              ["Cons", num(p.consistency)],
              ["Balanced", num(p.balanced)],
              ["Mins P", num(p.minutes_prob)],
              ["xGI/90", num(p.xgi_p90)],
              ["DEFCON/90", num(p.defcon_p90)],
            ].map(([k, v]) => (
              <div key={k} className="card rounded-lg px-3 py-2">
                <dt className="text-[11px] font-medium text-mute">{k}</dt>
                <dd className="tabular text-lg">{v}</dd>
              </div>
            ))}
          </dl>
          <p className="mt-3 text-sm text-mute">{p.fixture_run}</p>
        </div>
      </div>

      {history.length > 0 && (
        <section className="card rounded-xl p-4">
          <h2 className="mb-3 text-sm font-semibold text-mute">Points By GW</h2>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={history}>
                <XAxis dataKey="gw" stroke="#8b9bb4" fontSize={11} />
                <YAxis stroke="#8b9bb4" fontSize={11} />
                <Tooltip
                  contentStyle={{ background: "#121821", border: "1px solid #243042" }}
                />
                <Line type="monotone" dataKey="points" stroke="#3ee0a2" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </section>
      )}

      <section className="card overflow-x-auto rounded-xl p-4">
        <h2 className="mb-3 text-sm font-semibold text-mute">Remaining Fixtures</h2>
        <table className="w-full min-w-[480px] text-left text-sm">
          <thead className="text-xs text-mute">
            <tr>
              <th className="pb-2">GW</th>
              <th>Opp</th>
              <th>H/A</th>
              <th>FDR</th>
            </tr>
          </thead>
          <tbody>
            {data.fixtures.map((fx, i) => (
              <tr key={i} className="border-t border-line">
                <td className="py-1.5">{String(fx.event ?? "—")}</td>
                <td>{String(fx.opponent || "—")}</td>
                <td>{fx.is_home ? "H" : "A"}</td>
                <td>{String(fx.difficulty ?? "—")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  );
}
