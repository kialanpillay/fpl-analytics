import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../lib/api";
import { money, num, signed } from "../lib/format";
import { Pitch } from "../components/Pitch";
import { ModeSelector } from "../components/ModeSelector";
import { MODE_BLURB, type ObjectiveMode } from "../lib/modes";
import { useState } from "react";

export function Transfers() {
  const queryClient = useQueryClient();
  const [mode, setMode] = useState<ObjectiveMode>("balanced");
  const { data, isPending, isError, error } = useQuery({
    queryKey: ["transfers", mode],
    queryFn: () => (mode === "balanced" ? api.transfers() : api.solveTransfers({ objective: mode })),
  });
  const apply = useMutation({
    mutationFn: () =>
      api.applyTransfers(data?.plan?.players.map((p) => ({ id: p.id, name: p.web_name }))),
    onSuccess: () => queryClient.invalidateQueries(),
  });
  if (isError) return <p className="text-rose-300">{(error as Error).message}</p>;
  if (isPending || !data) return <p className="text-mute">Loading Transfers…</p>;
  const swaps = data.swaps ?? [];
  const nOut = data.n_transfers ?? swaps.length;

  return (
    <div className="space-y-6">
      <section className="card rounded-xl p-4">
        <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
          <h1 className="text-lg font-semibold">N-Transfer Plan</h1>
          <p className="text-sm text-mute">
            {nOut} Out · {data.hits} Hit Pts · Horizon Lift {signed(data.horizon_lift)}
          </p>
        </div>
        <ModeSelector value={mode} onChange={setMode} />
        <p className="mb-3 mt-2 text-sm text-mute">
          {MODE_BLURB[mode]}
        </p>
        {nOut === 0 ? (
          <p className="text-mute">No Improving Set Under Current Constraints.</p>
        ) : (
          <ul className="mb-4 space-y-2 text-sm">
            {swaps.map((swap, i) => {
              const out = swap.out;
              const inn = swap.in;
              return (
                <li key={`${out?.id ?? "out"}-${inn?.id ?? "in"}-${i}`}>
                  <span className="text-rose-300">{out?.web_name ?? "—"}</span> {out ? money(out.price) : ""}
                  {" → "}
                  <span className="text-emerald-300">{inn?.web_name ?? "—"}</span> {inn?.team_short}{" "}
                  {inn ? money(inn.price) : ""}
                  {out && inn && (
                    <span className="text-mute">
                      {" "}
                      (xH {num(inn.xp_horizon)} vs {num(out.xp_horizon)})
                    </span>
                  )}
                </li>
              );
            })}
          </ul>
        )}
        <div className="mb-4 flex gap-2 text-xs">
          {data.hit_table.map((row) => (
            <div key={row.hits} className="rounded border border-line px-2 py-1">
              {row.hits} Hits · −{row.cost} · Net {signed(row.net_horizon)}
            </div>
          ))}
        </div>
        {data.plan && (
          <>
            <button
              type="button"
              disabled={apply.isPending}
              onClick={() => apply.mutate()}
              className="mb-4 rounded-md bg-accent px-3 py-2 text-sm font-semibold text-ink"
            >
              {apply.isPending ? "Writing YAML…" : "Apply Plan To Squad File"}
            </button>
            <Pitch players={data.plan.players} xiIds={data.plan.xi_ids} benchIds={data.plan.bench_ids} />
          </>
        )}
        {apply.isError && <p className="mt-2 text-sm text-rose-300">{(apply.error as Error).message}</p>}
      </section>

      <section className="card overflow-x-auto rounded-xl p-4">
        <h2 className="mb-3 text-lg font-semibold">Legal 1-For-1</h2>
        <table className="w-full min-w-[720px] text-left text-sm">
          <thead className="text-xs text-mute">
            <tr>
              {["Out", "In", "Pos", "£", "Δ Bal", "Δ xH", "Δ PPP", "Own", "Uno"].map((h) => (
                <th key={h} className="pb-2 pr-3 font-medium">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.one_for_one.map((row) => (
              <tr key={`${row.out_id}-${row.in_id}`} className="border-t border-line">
                <td className="py-2 pr-3">{row.out}</td>
                <td className="pr-3 text-emerald-300">{row.in}</td>
                <td className="pr-3 text-mute">{row.position}</td>
                <td className="tabular pr-3">{signed(row.cost_delta, 1)}</td>
                <td className="tabular pr-3">{signed(row.d_balanced)}</td>
                <td className="tabular pr-3">{signed(row.d_xp)}</td>
                <td className="tabular pr-3">{signed(row.d_ppp)}</td>
                <td className="tabular pr-3">{num(row.in_own, 1)}</td>
                <td>{row.unorthodox ? "●" : ""}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  );
}
