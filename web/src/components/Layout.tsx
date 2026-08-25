import { NavLink, Outlet } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import { countdown, deadlineLabel } from "../lib/format";
import { useEffect, useState } from "react";

const NAV = [
  ["/", "Command"],
  ["/transfers", "Transfers"],
  ["/captaincy", "Captaincy"],
  ["/players", "Players"],
  ["/wildcard", "Wildcard"],
  ["/fixtures", "Fixtures"],
  ["/live", "Live"],
  ["/settings", "Settings"],
];

export function Layout() {
  const analysis = useQuery({ queryKey: ["analysis"], queryFn: api.analysis });
  const [tick, setTick] = useState(0);
  useEffect(() => {
    const id = window.setInterval(() => setTick((n) => n + 1), 30_000);
    return () => window.clearInterval(id);
  }, []);
  const meta = analysis.data?.meta;
  void tick;

  return (
    <div className="min-h-screen bg-ink">
      <header className="sticky top-0 z-20 border-b border-line bg-ink/90 backdrop-blur">
        <div className="mx-auto flex max-w-[1400px] flex-wrap items-center justify-between gap-3 px-4 py-3">
          <div>
            <p className="text-[11px] font-semibold tracking-[0.14em] text-accent">FPL Analytics</p>
            <p className="text-sm text-mute">
              {meta ? (
                <>
                  GW{meta.next_event} · {deadlineLabel(meta.deadline)} · {countdown(meta.deadline)}
                </>
              ) : (
                "Loading Season…"
              )}
            </p>
          </div>
          <div className="text-right text-xs text-mute">
            {analysis.data && (
              <>
                Bank £{analysis.data.meta.bank.toFixed(1)} · {analysis.data.meta.free_transfers} FTs
                <br />
                Fetched {new Date(analysis.data.fetched_at).toLocaleTimeString()}
              </>
            )}
          </div>
        </div>
        <nav className="mx-auto flex max-w-[1400px] gap-1 overflow-x-auto px-4 pb-2">
          {NAV.map(([to, label]) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              className={({ isActive }) =>
                `rounded-md px-3 py-1.5 text-sm ${isActive ? "bg-white/10 text-white" : "text-mute hover:text-white"}`
              }
            >
              {label}
            </NavLink>
          ))}
        </nav>
      </header>
      <main className="mx-auto max-w-[1400px] px-4 py-6">
        {analysis.isError && (
          <p className="mb-4 rounded-lg border border-rose-500/40 bg-rose-500/10 px-3 py-2 text-sm text-rose-200">
            {(analysis.error as Error).message}
          </p>
        )}
        <Outlet />
      </main>
    </div>
  );
}
