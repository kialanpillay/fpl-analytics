import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../lib/api";
import type { Analysis, Player } from "../lib/types";
import { money } from "../lib/format";

const QUOTAS: Record<string, number> = { GKP: 2, DEF: 5, MID: 5, FWD: 3 };

export function SquadEditor({ analysis }: { analysis: Analysis }) {
  const queryClient = useQueryClient();
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<Player[]>(analysis.squad);
  const [bank, setBank] = useState(analysis.meta.bank);
  const [fts, setFts] = useState(analysis.meta.free_transfers);

  useEffect(() => {
    setSelected(analysis.squad);
    setBank(analysis.meta.bank);
    setFts(analysis.meta.free_transfers);
  }, [analysis]);

  const pool = useQuery({
    queryKey: ["players", "editor"],
    queryFn: () => api.players({ sort: "balanced", n: 600 }),
  });

  const matches = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q || !pool.data) return [];
    const taken = new Set(selected.map((p) => p.id));
    return pool.data.players
      .filter((p) => !taken.has(p.id) && `${p.web_name} ${p.team_short} ${p.full_name || ""}`.toLowerCase().includes(q))
      .slice(0, 8);
  }, [pool.data, query, selected]);

  const save = useMutation({
    mutationFn: () =>
      api.putSquad({
        players: selected.map((p) => ({ id: p.id, name: p.web_name })),
        bank,
        free_transfers: fts,
      }),
    onSuccess: (data) => {
      queryClient.setQueryData(["analysis"], data);
      queryClient.invalidateQueries();
    },
  });

  const counts = selected.reduce<Record<string, number>>((acc, p) => {
    acc[p.position] = (acc[p.position] || 0) + 1;
    return acc;
  }, {});
  const cost = selected.reduce((s, p) => s + p.price, 0);
  const clubs = selected.reduce<Record<string, number>>((acc, p) => {
    acc[p.team_short] = (acc[p.team_short] || 0) + 1;
    return acc;
  }, {});
  const illegal = Object.entries(clubs).filter(([, n]) => n > 3);
  const quotaOk = Object.entries(QUOTAS).every(([pos, n]) => (counts[pos] || 0) === n);
  const legal = selected.length === 15 && quotaOk && cost <= 100 + 1e-6 && illegal.length === 0;

  return (
    <section className="card rounded-xl p-4">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-mute">Squad Editor</h2>
        <p className="text-xs text-mute">
          {selected.length}/15 · {money(cost)} · Writes config/squad.yaml
        </p>
      </div>
      <div className="mb-3 flex flex-wrap gap-2 text-xs">
        {Object.entries(QUOTAS).map(([pos, n]) => (
          <span key={pos} className={(counts[pos] || 0) === n ? "text-emerald-300" : "text-amber-300"}>
            {pos} {counts[pos] || 0}/{n}
          </span>
        ))}
        {illegal.map(([club, n]) => (
          <span key={club} className="text-rose-300">
            {club} {n}/3
          </span>
        ))}
      </div>
      <ul className="mb-3 max-h-56 space-y-1 overflow-auto text-sm">
        {selected.map((p) => (
          <li key={p.id} className="flex items-center justify-between gap-2">
            <span>
              {p.web_name} <span className="text-mute">{p.position} {p.team_short} {money(p.price)}</span>
            </span>
            <button
              type="button"
              className="text-xs text-rose-300"
              onClick={() => setSelected((cur) => cur.filter((x) => x.id !== p.id))}
            >
              Remove
            </button>
          </li>
        ))}
      </ul>
      <input
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Add Player…"
        className="mb-2 w-full rounded border border-line bg-ink px-3 py-2 text-sm"
      />
      {matches.length > 0 && (
        <ul className="mb-3 space-y-1 text-sm">
          {matches.map((p) => (
            <li key={p.id}>
              <button
                type="button"
                className="text-left text-accent"
                onClick={() => {
                  setSelected((cur) => [...cur, p]);
                  setQuery("");
                }}
              >
                {p.web_name} {p.position} {p.team_short} {money(p.price)}
              </button>
            </li>
          ))}
        </ul>
      )}
      <div className="flex flex-wrap items-end gap-3">
        <label className="text-xs text-mute">
          Bank
          <input
            type="number"
            step={0.1}
            className="mt-1 block w-20 rounded border border-line bg-ink px-2 py-1 text-sm"
            value={bank}
            onChange={(e) => setBank(Number(e.target.value))}
          />
        </label>
        <label className="text-xs text-mute">
          FTs
          <input
            type="number"
            className="mt-1 block w-16 rounded border border-line bg-ink px-2 py-1 text-sm"
            value={fts}
            onChange={(e) => setFts(Number(e.target.value))}
          />
        </label>
        <button
          type="button"
          disabled={!legal || save.isPending}
          onClick={() => save.mutate()}
          className="rounded-md bg-accent px-3 py-2 text-sm font-semibold text-ink disabled:opacity-50"
        >
          {save.isPending ? "Saving…" : "Save Squad"}
        </button>
        {!legal && <span className="text-xs text-amber-300">Fix Quotas, Club Cap, or Budget Before Save.</span>}
        {save.isError && <span className="text-xs text-rose-300">{(save.error as Error).message}</span>}
      </div>
    </section>
  );
}
