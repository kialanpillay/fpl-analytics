export type Player = {
  id: number;
  web_name: string;
  first_name?: string;
  second_name?: string;
  full_name?: string;
  position: "GKP" | "DEF" | "MID" | "FWD" | string;
  element_type?: number;
  team_id?: number;
  team?: string;
  team_short: string;
  team_code?: number;
  price: number;
  ownership?: number;
  status?: string;
  news?: string;
  chance_next?: number | null;
  event_points?: number;
  total_points?: number;
  xp_gw?: number;
  xp_horizon?: number;
  ppp?: number;
  consistency?: number;
  balanced?: number;
  residual?: number;
  differential?: number;
  minutes_prob?: number;
  role_risk?: number;
  effective_minutes?: number;
  next_fixture?: string;
  fixture_run?: string;
  fdr_mean?: number;
  unorthodox?: boolean;
  value_flag?: boolean;
  xg_p90?: number;
  xa_p90?: number;
  xgi_p90?: number;
  defcon_p90?: number;
  set_piece?: boolean;
  penalties_order?: number | null;
  minutes?: number;
  starts?: number;
  form?: number;
  bonus?: number;
  bps?: number;
  gw_minutes?: number;
  photo_url?: string | null;
  shirt_url?: string | null;
  badge_url?: string | null;
  captain_score?: number;
  captain_ev?: number;
  vice_ev?: number;
  season_default?: boolean;
  rank?: number;
  role?: string;
};

export type Note = { name: string; tone: string; note: string };
export type Strategy = { id: string; title: string; detail: string };

export type Meta = {
  next_event: number;
  current_event: number | null;
  deadline: string | null;
  total_managers: number;
  budget: number;
  team_limit: number;
  squad_size: number;
  season_started: boolean;
  horizon: number;
  bank: number;
  free_transfers: number;
};

export type Plan = {
  objective: string;
  cost: number;
  xp_gw: number;
  xp_horizon: number;
  ppp: number;
  consistency: number;
  xi: string[];
  bench: string[];
  xi_ids: number[];
  bench_ids: number[];
  players: Player[];
};

export type Analysis = {
  fetched_at: string;
  meta: Meta;
  squad: Player[];
  squad_eval: {
    n: number;
    cost: number;
    xp_gw: number;
    xp_horizon: number;
    ppp: number;
    consistency: number;
    balanced: number;
    club_counts: Record<string, number>;
    pos_counts: Record<string, number>;
    illegal_clubs: string[];
    dead_slots: string[];
    risk_names: string[];
  };
  notes: Note[];
  transfers: TransferRow[];
  transfer_plan: Plan | null;
  plans: Record<string, Plan>;
  underpriced: Player[];
  unorthodox: Player[];
  leaders: Record<string, Player[]>;
  strategies: Strategy[];
  xi_ids: number[];
  bench_ids: number[];
  warnings: string[];
};

export type TransferRow = {
  out_id: number;
  out: string;
  out_team?: string;
  in_id: number;
  in: string;
  in_team?: string;
  position: string;
  out_price?: number;
  in_price?: number;
  cost_delta: number;
  d_balanced: number;
  d_xp: number;
  d_ppp?: number;
  d_cons?: number;
  in_own?: number;
  unorthodox?: boolean;
};

export type TransferSwap = { out: Player | null; in: Player | null };

export type TransfersPayload = {
  one_for_one: TransferRow[];
  plan: Plan | null;
  incoming: Player[];
  outgoing: Player[];
  swaps: TransferSwap[];
  n_transfers: number;
  free_transfers: number;
  bank: number;
  hits: number;
  horizon_lift: number;
  hit_table: { hits: number; cost: number; net_horizon: number }[];
};

export type CaptaincyPayload = {
  recommended: Player | null;
  vice: Player | null;
  options: Player[];
};

export type FixtureCell = {
  team_id: number;
  event: number;
  fdr: number;
  home: boolean;
  opponent: string;
  kickoff: string | null;
  finished: boolean;
};

export type FixturesPayload = {
  next_event: number;
  events: number[];
  teams: {
    team_id: number;
    team: string;
    team_short: string;
    team_code: number;
    badge_url: string | null;
  }[];
  cells: FixtureCell[];
};

export type LivePayload = {
  event: number;
  deadline: string | null;
  status: unknown;
  squad: Player[];
  xi_ids: number[];
  bench_ids: number[];
  captain_id?: number | null;
  vice_id?: number | null;
  source: "entry" | "model";
  manager_id?: number | null;
  official: {
    points: number | null;
    points_on_bench: number | null;
    total_points: number | null;
    rank: number | null;
    overall_rank: number | null;
    transfers: number | null;
    hits: number | null;
    chip: string | null;
    auto_subs: { out_id: number; in_id: number }[];
  } | null;
  notes: Note[];
};

export type PlayerDetail = {
  player: Player;
  note: { tone: string; note: string } | null;
  in_squad: boolean;
  history: Array<Record<string, number | string | null>>;
  fixtures: Array<Record<string, number | string | boolean | null>>;
  history_past: Array<Record<string, number | string | null>>;
};

export type Settings = {
  manager_id: number | null;
  horizon: number;
  bank: number;
  free_transfers: number;
  budget: number;
  squad_path: string;
};

export type RunBody = {
  horizon?: number;
  refresh?: boolean;
  max_transfers?: number;
  objectives?: string[];
  bank?: number;
  free_transfers?: number;
};
