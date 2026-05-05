import { MacroTimeframes, TimeframeCode } from "../lib/macroTypes";

interface Props {
  timeframes: MacroTimeframes;
  selected: TimeframeCode;
  onChange: (tf: TimeframeCode) => void;
}

const ORDER: TimeframeCode[] = ["ytd", "6m", "1y", "2y", "max"];
const LABELS: Record<TimeframeCode, string> = {
  ytd: "YTD",
  "6m": "6M",
  "1y": "1Y",
  "2y": "2Y",
  max: "MAX",
};

/**
 * Server-precomputed timeframe toggle. Each chip swaps the entire active
 * regression bundle (raw / v1 / v2, scenarios, stock_betas, comparison) —
 * no client-side regression, all state changes are O(1).
 */
export function TimeframeSelector({ timeframes, selected, onChange }: Props) {
  const active = timeframes.by_timeframe[selected];
  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap",
      background: "#fff", border: "1px solid var(--border)",
      borderRadius: 8, padding: "8px 12px", marginBottom: 8,
    }}>
      <span style={{ fontSize: 11, fontWeight: 600, color: "var(--muted)",
                     textTransform: "uppercase", letterSpacing: 0.5 }}>
        Timeframe
      </span>
      <div style={{ display: "flex", gap: 4 }}>
        {ORDER.map((tf) => {
          const available = !!timeframes.by_timeframe[tf];
          const isActive = tf === selected;
          return (
            <button
              key={tf}
              disabled={!available}
              onClick={() => onChange(tf)}
              style={{
                padding: "5px 14px",
                fontSize: 12, fontWeight: 600,
                border: "1px solid var(--border)",
                borderRadius: 4,
                background: isActive ? "var(--accent)" : "#fff",
                color: isActive ? "#fff" : (available ? "var(--text)" : "var(--muted)"),
                cursor: available ? "pointer" : "not-allowed",
                opacity: available ? 1 : 0.4,
              }}
              title={available
                ? `${LABELS[tf]} · ${timeframes.by_timeframe[tf].n_obs} obs`
                : "insufficient data for this timeframe"}
            >
              {LABELS[tf]}
            </button>
          );
        })}
      </div>
      {active && (
        <small style={{ color: "var(--muted)", marginLeft: "auto" }}>
          {active.date_range[0]} → {active.date_range[1]} · n = {active.n_obs} obs ·
          R² = {active.v2.r_squared.toFixed(3)}
        </small>
      )}
    </div>
  );
}
