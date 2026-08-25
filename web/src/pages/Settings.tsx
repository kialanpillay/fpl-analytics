import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../lib/api";

export function Settings() {
  const queryClient = useQueryClient();
  const { data, isPending } = useQuery({ queryKey: ["settings"], queryFn: api.settings });
  const [horizon, setHorizon] = useState<number>(6);

  useEffect(() => {
    if (!data) return;
    setHorizon(data.horizon);
  }, [data]);

  const save = useMutation({
    mutationFn: () => api.putSettings({ horizon }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["settings"] });
    },
  });
  const importEntry = useMutation({
    mutationFn: () => api.importEntry(data!.manager_id as number),
    onSuccess: () => queryClient.invalidateQueries(),
  });

  if (isPending || !data) return <p className="text-mute">Loading Settings…</p>;

  return (
    <div className="card max-w-xl space-y-4 rounded-xl p-5">
      <h1 className="text-lg font-semibold">Settings</h1>
      <p className="text-sm text-mute">
        Squad File: <span className="text-slate-200">{data.squad_path}</span>
        . Import Picks Overwrites It From The Official 15.
      </p>
      <label className="block text-sm text-mute">
        Default Horizon
        <span className="mt-0.5 block font-normal text-mute/80">
          Gameweeks Ahead for xp_horizon, FDR run, and transfer lift. Re-Run can override.
        </span>
        <input
          type="number"
          min={1}
          max={10}
          className="mt-1 block w-24 rounded border border-line bg-ink px-3 py-2 text-white"
          value={horizon}
          onChange={(e) => setHorizon(Number(e.target.value))}
        />
      </label>
      <div className="flex flex-wrap gap-2">
        <button
          type="button"
          className="rounded-md bg-accent px-3 py-2 text-sm font-semibold text-ink"
          onClick={() => save.mutate()}
        >
          Save Settings
        </button>
        <button
          type="button"
          disabled={!data.manager_id}
          className="rounded-md border border-line px-3 py-2 text-sm"
          onClick={() => importEntry.mutate()}
        >
          {importEntry.isPending ? "Importing…" : "Import Picks"}
        </button>
      </div>
      {save.isSuccess && <p className="text-sm text-emerald-300">Saved.</p>}
      {importEntry.isSuccess && (
        <p className="text-sm text-emerald-300">
          Imported {importEntry.data.team_name} ({importEntry.data.player_name})
        </p>
      )}
      {(save.isError || importEntry.isError) && (
        <p className="text-sm text-rose-300">
          {((save.error || importEntry.error) as Error).message}
        </p>
      )}
    </div>
  );
}
