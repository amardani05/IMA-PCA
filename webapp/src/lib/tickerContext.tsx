import { createContext, useContext } from "react";

/** App-level "open the single-name drawer" action, provided by App. */
export const TickerOpenContext = createContext<(ticker: string) => void>(() => {});

export function useOpenTicker() {
  return useContext(TickerOpenContext);
}

/** Clickable ticker cell — opens the research drawer for the name. */
export function TickerLink({ ticker, children }: { ticker: string; children?: React.ReactNode }) {
  const open = useOpenTicker();
  return (
    <button
      onClick={(e) => { e.stopPropagation(); open(ticker); }}
      title={`Open ${ticker} detail`}
      style={{
        background: "none", border: "none", padding: 0, cursor: "pointer",
        color: "var(--accent)", font: "inherit", fontWeight: 700,
        textDecoration: "none",
      }}>
      {children ?? ticker}
    </button>
  );
}
