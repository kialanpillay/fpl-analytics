import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import { Pitch } from "../components/Pitch";
import { money, num } from "../lib/format";
import type { Plan } from "../lib/types";

const MODES = [
  ["balanced", "Balanced"],
  ["ppp", "PPP"],
  ["consistency", "Consistency"],
  ["differential", "Differential"],
] as const;

export function Drafts() {
  const { data, isPending } = useQuery({ queryKey: ["drafts"], queryFn: api.drafts });
  const analysis = useQuery({ queryKey: ["analysis"], queryFn: api.analysis });
  const [mode, setMode] = useState<(typeof MODES)[number][0]>("balanced");
  const [locked, setLocked] = useState<number[]>([]);
  const [banned, setBanned] = useState<number[]>([]);
  const solve = useMutation({
    mutationFn: () => api.solveDraft({ objective: mode, locked_ids: locked, banned_ids: banned }),
  });

  if (isPending || !data) return <p className="text-mute">Solving Drafts…</p>;
  const plan: Plan | undefined = solve.data?.plan || data.plans[mode];

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap gap-2">
        {MODES.map(([value, label]) => (
          <button
            key={value}
            type="button"
            onClick={() => setMode(value)}
            className={`rounded-md px-3 py-1.5 text-sm ${mode === value ? "bg-accent text-ink" : "bg-white/5 text-mute"}`}
          >
            {label}
          </button>
        ))}
      </div>
      {plan && (
        <p className="text-sm text-mute">
          {money(plan.cost)} · XI xPts {num(plan.xp_gw, 1)} · Horizon {num(plan.xp_horizon, 1)} · PPP {num(plan.ppp)}
        </p>
      )}
      {analysis.data && (
        <div className="card rounded-xl p-4 text-sm">
          <p className="mb-2 text-xs font-medium text-mute">Locks / Bans (From Current 15)</p>
          <div className="flex flex-wrap gap-2">
            {analysis.data.squad.map((p) => {
              const isLocked = locked.includes(p.id);
              const isBanned = banned.includes(p.id);
              return (
                <button
                  key={p.id}
                  type="button"
                  onClick={() => {
                    if (isLocked) {
                      setLocked((xs) => xs.filter((id) => id !== p.id));
                      setBanned((xs) => [...xs, p.id]);
                    } else if (isBanned) {
                      setBanned((xs) => xs.filter((id) => id !== p.id));
                    } else {
                      setLocked((xs) => [...xs, p.id]);
                    }
                  }}
                  className={`rounded border px-2 py-1 ${
                    isLocked
                      ? "border-emerald-400 text-emerald-300"
                      : isBanned
                        ? "border-rose-400 text-rose-300"
                        : "border-line text-mute"
                  }`}
                >
                  {p.web_name}
                </button>
              );
            })}
          </div>
          <button
            type="button"
            className="mt-3 rounded-md bg-accent px-3 py-1.5 text-sm font-semibold text-ink"
            onClick={() => solve.mutate()}
          >
            {solve.isPending ? "Solving…" : "Re-Solve With Locks"}
          </button>
          {solve.isError && <p className="mt-2 text-rose-300">{(solve.error as Error).message}</p>}
        </div>
      )}
      {plan ? (
        <Pitch players={plan.players} xiIds={plan.xi_ids} benchIds={plan.bench_ids} />
      ) : (
        <p className="text-amber-300">No Plan For {mode}. Is Pulp Installed?</p>
      )}
      {solve.data && (
        <ul className="space-y-1 text-sm text-mute">
          {(solve.data.swaps?.length ? solve.data.swaps : []).map((swap, i) => (
            <li key={`${swap.out?.id ?? "out"}-${swap.in?.id ?? "in"}-${i}`}>
              <span className="text-rose-300">{swap.out?.web_name ?? "—"}</span>
              {" → "}
              <span className="text-emerald-300">{swap.in?.web_name ?? "—"}</span>
            </li>
          ))}
          {!solve.data.swaps?.length && (
            <li>
              In: {solve.data.incoming.map((p) => p.web_name).join(", ") || "—"} · Out:{" "}
              {solve.data.outgoing.map((p) => p.web_name).join(", ") || "—"}
            </li>
          )}
        </ul>
      )}
    </div>
  );
}
