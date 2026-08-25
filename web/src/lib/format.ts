export function money(n: number | null | undefined): string {
  if (n == null || Number.isNaN(n)) return "—";
  return `£${n.toFixed(1)}`;
}

export function num(n: number | null | undefined, digits = 2): string {
  if (n == null || Number.isNaN(n)) return "—";
  return n.toFixed(digits);
}

export function signed(n: number | null | undefined, digits = 2): string {
  if (n == null || Number.isNaN(n)) return "—";
  const prefix = n > 0 ? "+" : "";
  return `${prefix}${n.toFixed(digits)}`;
}

export function fdrFromFixture(label?: string | null): number | null {
  if (!label) return null;
  const match = label.match(/\((\d)\)\s*$/);
  return match ? Number(match[1]) : null;
}

export function deadlineLabel(iso: string | null | undefined): string {
  if (!iso) return "No Deadline";
  const when = new Date(iso);
  if (Number.isNaN(when.getTime())) return iso;
  return when.toLocaleString(undefined, {
    weekday: "short",
    day: "numeric",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function countdown(iso: string | null | undefined): string {
  if (!iso) return "—";
  const ms = new Date(iso).getTime() - Date.now();
  if (Number.isNaN(ms)) return "—";
  if (ms <= 0) return "Deadline Passed";
  const hours = Math.floor(ms / 3_600_000);
  const days = Math.floor(hours / 24);
  const remH = hours % 24;
  const mins = Math.floor((ms % 3_600_000) / 60_000);
  if (days > 0) return `${days}d ${remH}h`;
  if (hours > 0) return `${hours}h ${mins}m`;
  return `${mins}m`;
}

export function toneClass(tone: string): string {
  if (tone === "value") return "text-emerald-300";
  if (tone === "watch") return "text-amber-300";
  if (tone === "risk") return "text-orange-300";
  if (tone === "avoid") return "text-rose-300";
  return "text-slate-300";
}

export function statusDot(status?: string): string {
  if (status === "a") return "bg-emerald-400";
  if (status === "d") return "bg-amber-400";
  if (status === "i") return "bg-rose-500";
  return "bg-slate-500";
}
