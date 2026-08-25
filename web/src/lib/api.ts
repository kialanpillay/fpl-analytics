import type {
  Analysis,
  CaptaincyPayload,
  FixturesPayload,
  LivePayload,
  Player,
  PlayerDetail,
  RunBody,
  Settings,
  TransfersPayload,
} from "./types";

const DEFAULT_TIMEOUT_MS = 180_000;

async function request<T>(path: string, init?: RequestInit, timeoutMs = DEFAULT_TIMEOUT_MS): Promise<T> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(path, {
      headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
      signal: controller.signal,
      ...init,
    });
    if (!res.ok) {
      let detail = res.statusText;
      try {
        const body = await res.json();
        detail = body.detail || JSON.stringify(body);
      } catch {
        detail = await res.text();
      }
      throw new Error(detail);
    }
    return res.json() as Promise<T>;
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new Error("Request Timed Out. The Pipeline Can Take a Minute on First Load.");
    }
    throw err;
  } finally {
    window.clearTimeout(timer);
  }
}

export const api = {
  health: () => request<{ ok: boolean; version: string; fetched_at: string | null; pulp: boolean }>("/api/health"),
  analysis: () => request<Analysis>("/api/analysis"),
  run: (body: RunBody) =>
    request<Analysis>("/api/analysis/run", { method: "POST", body: JSON.stringify(body) }),
  squad: () => request("/api/squad"),
  putSquad: (body: { players: { id: number; name?: string }[]; bank?: number; free_transfers?: number }) =>
    request<Analysis>("/api/squad", { method: "PUT", body: JSON.stringify(body) }),
  importEntry: (manager_id: number) =>
    request<{ manager_id: number; team_name: string; player_name: string; analysis: Analysis }>(
      "/api/squad/import-entry",
      { method: "POST", body: JSON.stringify({ manager_id }) },
    ),
  transfers: () => request<TransfersPayload>("/api/transfers"),
  applyTransfers: () =>
    request<Analysis>("/api/transfers/apply", { method: "POST", body: JSON.stringify({ use_plan: true }) }),
  captaincy: () => request<CaptaincyPayload>("/api/captaincy"),
  players: (q: { position?: string; sort?: string; n?: number } = {}) => {
    const params = new URLSearchParams();
    if (q.position) params.set("position", q.position);
    if (q.sort) params.set("sort", q.sort);
    if (q.n) params.set("n", String(q.n));
    return request<{ players: Player[]; sort: string; n: number }>(`/api/players?${params}`);
  },
  player: (id: number) => request<PlayerDetail>(`/api/players/${id}`),
  drafts: () => request<{ plans: Analysis["plans"]; pulp: boolean }>("/api/drafts"),
  solveDraft: (body: { objective: string; locked_ids: number[]; banned_ids: number[]; budget?: number }) =>
    request<{ plan: Analysis["plans"][string]; incoming: Player[]; outgoing: Player[]; swaps: { out: Player | null; in: Player | null }[] }>(
      "/api/drafts/solve",
      { method: "POST", body: JSON.stringify(body) },
    ),
  fixtures: (horizon?: number) =>
    request<FixturesPayload>(`/api/fixtures${horizon ? `?horizon=${horizon}` : ""}`),
  live: (refresh = false) => request<LivePayload>(`/api/live${refresh ? "?refresh=true" : ""}`),
  settings: () => request<Settings>("/api/settings"),
  putSettings: (body: { horizon?: number }) =>
    request<Settings>("/api/settings", { method: "PUT", body: JSON.stringify(body) }),
};
