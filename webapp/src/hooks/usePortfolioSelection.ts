import { useCallback, useMemo, useState } from "react";
import { recomputePortfolioBetas } from "../lib/macro";
import { StockBetaMatrix } from "../lib/macroTypes";
import { PortfolioRow, UniverseRow } from "../lib/types";

export interface SelectionState {
  selectedTickers: Set<string>;
  weights: Record<string, number>;
  effectiveWeights: Record<string, number>;
  totalWeight: number;
  liveBetas: Record<string, number>;
  isModified: boolean;

  toggleStock: (ticker: string) => void;
  toggleSector: (sector: string) => void;
  selectAll: () => void;
  clearAll: () => void;
  resetToDefault: () => void;
  addCandidate: (ticker: string, weight?: number) => void;
}

interface Args {
  defaultPortfolio: PortfolioRow[];   // current IMA portfolio (anchor for "default")
  universe?: UniverseRow[];           // for sector membership of universe stocks
  stockBetas: StockBetaMatrix | null;
}

export function usePortfolioSelection({ defaultPortfolio, universe, stockBetas }: Args): SelectionState {
  const defaultWeights = useMemo(() => {
    const w: Record<string, number> = {};
    for (const p of defaultPortfolio) {
      if (p.Weight && p.Weight > 0) w[p.Ticker] = p.Weight;
    }
    return w;
  }, [defaultPortfolio]);

  const defaultSet = useMemo(() => new Set(Object.keys(defaultWeights)), [defaultWeights]);

  const [weights, setWeights] = useState<Record<string, number>>(defaultWeights);
  const [selectedTickers, setSelectedTickers] = useState<Set<string>>(new Set(defaultSet));

  const sectorByTicker = useMemo(() => {
    const m: Record<string, string> = {};
    for (const p of defaultPortfolio) {
      if (p.Sector) m[p.Ticker] = p.Sector;
    }
    if (universe) {
      for (const u of universe) {
        if (!m[u.Ticker] && u.Sector) m[u.Ticker] = u.Sector;
      }
    }
    return m;
  }, [defaultPortfolio, universe]);

  const toggleStock = useCallback((ticker: string) => {
    setSelectedTickers((prev) => {
      const next = new Set(prev);
      if (next.has(ticker)) next.delete(ticker);
      else next.add(ticker);
      return next;
    });
  }, []);

  const toggleSector = useCallback((sector: string) => {
    setSelectedTickers((prev) => {
      const inSector = Object.entries(sectorByTicker)
        .filter(([_, s]) => s === sector)
        .map(([t]) => t)
        .filter((t) => weights[t] !== undefined);
      const allSelected = inSector.every((t) => prev.has(t));
      const next = new Set(prev);
      for (const t of inSector) {
        if (allSelected) next.delete(t);
        else next.add(t);
      }
      return next;
    });
  }, [sectorByTicker, weights]);

  const selectAll = useCallback(() => {
    setSelectedTickers(new Set(Object.keys(weights)));
  }, [weights]);

  const clearAll = useCallback(() => {
    setSelectedTickers(new Set());
  }, []);

  const resetToDefault = useCallback(() => {
    setWeights(defaultWeights);
    setSelectedTickers(new Set(defaultSet));
  }, [defaultWeights, defaultSet]);

  const addCandidate = useCallback((ticker: string, weight: number = 0.02) => {
    setWeights((prev) => ({ ...prev, [ticker]: weight }));
    setSelectedTickers((prev) => new Set([...prev, ticker]));
  }, []);

  const totalWeight = useMemo(() => {
    let s = 0;
    selectedTickers.forEach((t) => { s += weights[t] ?? 0; });
    return s;
  }, [selectedTickers, weights]);

  const { betas: liveBetas, effectiveWeights } = useMemo(() => {
    if (!stockBetas) return { betas: {}, effectiveWeights: {} };
    return recomputePortfolioBetas(selectedTickers, weights, stockBetas);
  }, [selectedTickers, weights, stockBetas]);

  const isModified = useMemo(() => {
    if (selectedTickers.size !== defaultSet.size) return true;
    for (const t of defaultSet) {
      if (!selectedTickers.has(t)) return true;
    }
    return false;
  }, [selectedTickers, defaultSet]);

  return {
    selectedTickers, weights, effectiveWeights, totalWeight,
    liveBetas, isModified,
    toggleStock, toggleSector, selectAll, clearAll, resetToDefault, addCandidate,
  };
}
