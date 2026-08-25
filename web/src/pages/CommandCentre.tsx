import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import { KpiStrip } from "../components/KpiStrip";
import { Notes } from "../components/Notes";
import { Pitch } from "../components/Pitch";
import { RerunPanel } from "../components/RerunPanel";
import { SquadEditor } from "../components/SquadEditor";

export function CommandCentre() {
  const { data, isPending, isError, error } = useQuery({ queryKey: ["analysis"], queryFn: api.analysis });
  const cap = useQuery({ queryKey: ["captaincy"], queryFn: api.captaincy });
  if (isError) return <p className="text-rose-300">{(error as Error).message}</p>;
  if (isPending || !data) return <p className="text-mute">Running Pipeline…</p>;

  return (
    <div className="space-y-5">
      <RerunPanel analysis={data} />
      <KpiStrip analysis={data} />
      {(data.squad_eval.illegal_clubs.length > 0 || data.squad_eval.dead_slots.length > 0) && (
        <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-sm text-amber-100">
          {data.squad_eval.illegal_clubs.length > 0 && <p>Illegal Club Cap: {data.squad_eval.illegal_clubs.join(", ")}</p>}
          {data.squad_eval.dead_slots.length > 0 && <p>Dead Slots: {data.squad_eval.dead_slots.join(", ")}</p>}
          {data.squad_eval.risk_names.length > 0 && <p>Role Risk: {data.squad_eval.risk_names.join(", ")}</p>}
        </div>
      )}
      <Pitch
        players={data.squad}
        xiIds={data.xi_ids}
        benchIds={data.bench_ids}
        captainId={cap.data?.recommended?.id}
        viceId={cap.data?.vice?.id}
      />
      <Notes notes={data.notes} strategies={data.strategies} warnings={data.warnings} />
      <SquadEditor analysis={data} />
    </div>
  );
}
