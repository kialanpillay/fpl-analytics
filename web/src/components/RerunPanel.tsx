import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "../lib/api";
import type { Analysis } from "../lib/types";

export function RerunPanel({ analysis }: { analysis: Analysis }) {
  const queryClient = useQueryClient();
  const [horizon, setHorizon] = useState(analysis.meta.horizon);
  const [maxTransfers, setMaxTransfers] = useState(2);
  const [refresh, setRefresh] = useState(false);
  const [bank, setBank] = useState(analysis.meta.bank);
  const [fts, setFts] = useState(analysis.meta.free_transfers);

  const run = useMutation({
    mutationFn: () =>
      api.run({
        horizon,
        refresh,
        max_transfers: maxTransfers,
        bank,
        free_transfers: fts,
      }),
    onSuccess: (data) => {
      queryClient.setQueryData(["analysis"], data);
      queryClient.invalidateQueries({ queryKey: ["transfers"] });
      queryClient.invalidateQueries({ queryKey: ["captaincy"] });
      queryClient.invalidateQueries({ queryKey: ["wildcard"] });
      queryClient.invalidateQueries({ queryKey: ["players"] });
      queryClient.invalidateQueries({ queryKey: ["live"] });
      queryClient.invalidateQueries({ queryKey: ["fixtures"] });
    },
  });

  return (
    <form
      className="card flex flex-wrap items-end gap-3 rounded-xl p-3"
      onSubmit={(e) => {
        e.preventDefault();
        run.mutate();
      }}
    >
      <label className="text-xs text-mute">
        Horizon
        <input
          type="number"
          min={1}
          max={10}
          className="mt-1 block w-20 rounded border border-line bg-ink px-2 py-1 text-sm text-white"
          value={horizon}
          onChange={(e) => setHorizon(Number(e.target.value))}
        />
      </label>
      <label className="text-xs text-mute">
        Max Transfers
        <input
          type="number"
          min={1}
          max={3}
          className="mt-1 block w-20 rounded border border-line bg-ink px-2 py-1 text-sm text-white"
          value={maxTransfers}
          onChange={(e) => setMaxTransfers(Number(e.target.value))}
        />
      </label>
      <label className="text-xs text-mute">
        Bank
        <input
          type="number"
          step={0.1}
          min={0}
          className="mt-1 block w-20 rounded border border-line bg-ink px-2 py-1 text-sm text-white"
          value={bank}
          onChange={(e) => setBank(Number(e.target.value))}
        />
      </label>
      <label className="text-xs text-mute">
        FTs
        <input
          type="number"
          min={0}
          max={5}
          className="mt-1 block w-16 rounded border border-line bg-ink px-2 py-1 text-sm text-white"
          value={fts}
          onChange={(e) => setFts(Number(e.target.value))}
        />
      </label>
      <label className="flex items-center gap-2 pb-2 text-xs text-mute">
        <input type="checkbox" checked={refresh} onChange={(e) => setRefresh(e.target.checked)} />
        Refresh Cache
      </label>
      <button
        type="submit"
        disabled={run.isPending}
        className="rounded-md bg-accent px-4 py-2 text-sm font-semibold text-ink disabled:opacity-60"
      >
        {run.isPending ? "Solving…" : "Re-Run"}
      </button>
      {run.isError && <p className="text-sm text-rose-300">{(run.error as Error).message}</p>}
      {run.isSuccess && <p className="text-sm text-emerald-300">Updated {new Date(run.data.fetched_at).toLocaleTimeString()}</p>}
    </form>
  );
}
