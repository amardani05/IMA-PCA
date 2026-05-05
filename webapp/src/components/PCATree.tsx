import { useMemo, useState } from "react";
import { PortfolioRow, UniverseRow } from "../lib/types";
import { tierClass, fmt } from "../lib/data";

interface Props {
  portfolio: PortfolioRow[];
  universe: UniverseRow[];
  selectedTickers: Set<string>;
  toggleStock: (ticker: string) => void;
  toggleSector: (sector: string) => void;
  selectAll: () => void;
  resetToPortfolio: () => void;
  clearAll: () => void;
}

interface SectorBlock {
  sector: string;
  holdings: PortfolioRow[];
  others: UniverseRow[];           // universe stocks in same sector NOT in portfolio
  totalHoldingsWeight: number;
  selectedHoldings: number;
  someSelected: boolean;
  allSelected: boolean;
}

/**
 * Stock selector laid out as a 4-column grid of sector tiles.
 *
 * Each tile shows ALL portfolio holdings for that sector (always visible) plus
 * a "+ N others" expand toggle that reveals universe peers in the same sector.
 * Selected tickers (in any tile) are highlighted on every PC chart simultaneously.
 */
export function PCATree(props: Props) {
  const { portfolio, universe, selectedTickers,
          toggleStock, toggleSector, selectAll, resetToPortfolio, clearAll } = props;

  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  const blocks: SectorBlock[] = useMemo(() => {
    const portfolioSet = new Set(portfolio.map((p) => p.Ticker));
    const sectorMap = new Map<string, { holdings: PortfolioRow[]; others: UniverseRow[] }>();

    for (const p of portfolio) {
      const sector = p.Sector ?? "Unknown";
      if (!sectorMap.has(sector)) sectorMap.set(sector, { holdings: [], others: [] });
      sectorMap.get(sector)!.holdings.push(p);
    }

    for (const u of universe) {
      if (portfolioSet.has(u.Ticker)) continue;
      if (!sectorMap.has(u.Sector)) continue;
      sectorMap.get(u.Sector)!.others.push(u);
    }
    for (const v of sectorMap.values()) {
      v.others.sort((a, b) => a.composite_score - b.composite_score);
    }

    const out: SectorBlock[] = [];
    for (const [sector, { holdings, others }] of sectorMap.entries()) {
      const totalHoldingsWeight = holdings.reduce((s, h) => s + (h.Weight ?? 0), 0);
      const selectedHoldings = holdings.filter((h) => selectedTickers.has(h.Ticker)).length;
      out.push({
        sector, holdings, others, totalHoldingsWeight, selectedHoldings,
        someSelected: selectedHoldings > 0 && selectedHoldings < holdings.length,
        allSelected: selectedHoldings === holdings.length && holdings.length > 0,
      });
    }
    out.sort((a, b) => b.totalHoldingsWeight - a.totalHoldingsWeight);
    return out;
  }, [portfolio, universe, selectedTickers]);

  const toggleExpand = (sector: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(sector)) next.delete(sector);
      else next.add(sector);
      return next;
    });
  };

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between",
                     alignItems: "center", marginBottom: 8, padding: "0 4px" }}>
        <div>
          <strong style={{ fontSize: 14 }}>Stock selector</strong>
          <small className="muted" style={{ marginLeft: 8 }}>
            {selectedTickers.size} selected · highlighted with black ring &amp; ticker label
          </small>
        </div>
        <div style={{ display: "flex", gap: 4 }}>
          <button onClick={selectAll} style={btnSm}>Portfolio</button>
          <button onClick={clearAll} style={btnSm}>None</button>
          <button onClick={resetToPortfolio} style={btnSm}>Reset</button>
        </div>
      </div>

      <div style={{
        display: "grid",
        gridTemplateColumns: "repeat(4, minmax(0, 1fr))",
        gap: 10,
      }}>
        {blocks.map((b) => (
          <SectorTile
            key={b.sector}
            block={b}
            isExpanded={expanded.has(b.sector)}
            selectedTickers={selectedTickers}
            onToggleStock={toggleStock}
            onToggleSector={toggleSector}
            onToggleExpand={toggleExpand}
          />
        ))}
      </div>
    </div>
  );
}


function SectorTile({
  block, isExpanded, selectedTickers,
  onToggleStock, onToggleSector, onToggleExpand,
}: {
  block: SectorBlock;
  isExpanded: boolean;
  selectedTickers: Set<string>;
  onToggleStock: (t: string) => void;
  onToggleSector: (s: string) => void;
  onToggleExpand: (s: string) => void;
}) {
  return (
    <div style={{
      background: "#fff",
      border: "1px solid var(--border)",
      borderRadius: 8,
      padding: "10px 12px",
      boxShadow: "0 1px 2px rgba(0,0,0,0.03)",
      display: "flex", flexDirection: "column",
    }}>
      <div style={{ display: "flex", alignItems: "center", marginBottom: 6,
                     paddingBottom: 6, borderBottom: "1px solid var(--border)" }}>
        <input type="checkbox"
               checked={block.allSelected}
               ref={(el) => { if (el) el.indeterminate = block.someSelected; }}
               onChange={() => onToggleSector(block.sector)}
               style={{ marginRight: 8 }} />
        <strong style={{ flex: 1, fontSize: 13, lineHeight: 1.2 }}>
          {block.sector}
        </strong>
        <small style={{ color: "var(--muted)", fontVariantNumeric: "tabular-nums" }}>
          {(block.totalHoldingsWeight * 100).toFixed(1)}%
        </small>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 1 }}>
        {block.holdings.map((h) => (
          <StockRow key={h.Ticker}
                    ticker={h.Ticker}
                    isPortfolio
                    weight={h.Weight}
                    riskTier={h.Risk_Tier}
                    score={h.Composite_Score ?? null}
                    isSelected={selectedTickers.has(h.Ticker)}
                    onToggle={() => onToggleStock(h.Ticker)} />
        ))}
      </div>

      {block.others.length > 0 && (
        <>
          <button
            onClick={() => onToggleExpand(block.sector)}
            style={{
              background: "none", border: "none", color: "var(--muted)",
              cursor: "pointer", fontSize: 11, padding: "6px 0 2px",
              display: "flex", alignItems: "center", gap: 4, marginTop: "auto",
              textAlign: "left",
            }}
          >
            <span style={{ display: "inline-block",
                            transform: isExpanded ? "rotate(90deg)" : "rotate(0deg)",
                            transition: "transform 0.15s" }}>▶</span>
            {isExpanded ? "Hide peers" : `+ ${block.others.length} peers in sector`}
          </button>
          {isExpanded && (
            <div style={{ borderLeft: "2px solid var(--border)",
                           paddingLeft: 6, marginTop: 2,
                           maxHeight: 180, overflowY: "auto" }}>
              {block.others.map((u) => (
                <StockRow key={u.Ticker}
                          ticker={u.Ticker}
                          isPortfolio={false}
                          weight={null}
                          riskTier={u.risk_tier}
                          score={u.composite_score}
                          isSelected={selectedTickers.has(u.Ticker)}
                          onToggle={() => onToggleStock(u.Ticker)} />
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}


function StockRow({
  ticker, isPortfolio, weight, riskTier, score, isSelected, onToggle,
}: {
  ticker: string;
  isPortfolio: boolean;
  weight: number | null;
  riskTier: string | undefined | null;
  score: number | null;
  isSelected: boolean;
  onToggle: () => void;
}) {
  return (
    <label style={{
      display: "flex", alignItems: "center", padding: "1px 2px",
      borderRadius: 3, fontSize: 12, cursor: "pointer",
      background: isSelected ? "#eef2fb" : "transparent",
    }}>
      <input type="checkbox" checked={isSelected} onChange={onToggle}
             style={{ marginRight: 5 }} />
      <strong style={{
        minWidth: 50, fontSize: 12,
        color: isPortfolio ? "var(--accent)" : "var(--text)",
      }}>{ticker}</strong>
      <span style={{ flex: 1 }} />
      {riskTier && (
        <span className={tierClass(riskTier)}
              style={{ fontSize: 9, padding: "1px 4px", marginRight: 4 }}>
          {riskTier}
        </span>
      )}
      <small style={{ color: "var(--muted)", fontVariantNumeric: "tabular-nums",
                       minWidth: 32, textAlign: "right" }}>
        {weight !== null
          ? `${(weight * 100).toFixed(1)}%`
          : score != null ? fmt(score, 0) : ""}
      </small>
    </label>
  );
}

const btnSm: React.CSSProperties = {
  padding: "4px 10px", fontSize: 12, border: "1px solid var(--border)",
  background: "#fff", borderRadius: 4, cursor: "pointer",
};
