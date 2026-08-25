export const OBJECTIVE_MODES = [
  ["balanced", "Balanced"],
  ["aggressive", "Aggressive"],
  ["template", "Template"],
  ["ppp", "PPP"],
  ["consistency", "Consistency"],
  ["differential", "Differential"],
] as const;

export type ObjectiveMode = (typeof OBJECTIVE_MODES)[number][0];

export const MODE_BLURB: Record<ObjectiveMode, string> = {
  balanced: "Points, Value, Floor.",
  aggressive: "Next GW + Attack. No Price or Floor.",
  template: "Points + Popularity.",
  ppp: "Horizon Points Per Million.",
  consistency: "Minutes and DEFCON.",
  differential: "Balanced × Low Ownership.",
};
