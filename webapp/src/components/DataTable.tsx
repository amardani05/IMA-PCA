import { useMemo, useState } from "react";

export type Column<T> = {
  key: string;
  header: string;
  accessor: (row: T) => any;
  render?: (row: T) => React.ReactNode;
  numeric?: boolean;
  defaultSortDesc?: boolean;
};

interface Props<T> {
  rows: T[];
  columns: Column<T>[];
  rowClassName?: (row: T) => string | undefined;
  initialSortKey?: string;
  stickyHeader?: boolean;
  pageSize?: number;
  filterText?: string;
  filterFn?: (row: T, text: string) => boolean;
}

export function DataTable<T>(props: Props<T>) {
  const { rows, columns, rowClassName, initialSortKey, pageSize,
          filterText = "", filterFn } = props;

  const [sortKey, setSortKey] = useState<string | null>(initialSortKey ?? null);
  const [sortDesc, setSortDesc] = useState<boolean>(true);
  const [page, setPage] = useState(0);

  const filtered = useMemo(() => {
    if (!filterText || !filterFn) return rows;
    const t = filterText.toLowerCase();
    return rows.filter((r) => filterFn(r, t));
  }, [rows, filterText, filterFn]);

  const sorted = useMemo(() => {
    if (!sortKey) return filtered;
    const col = columns.find((c) => c.key === sortKey);
    if (!col) return filtered;
    const copy = [...filtered];
    copy.sort((a, b) => {
      const av = col.accessor(a);
      const bv = col.accessor(b);
      if (av == null && bv == null) return 0;
      if (av == null) return 1;
      if (bv == null) return -1;
      if (typeof av === "number" && typeof bv === "number") {
        return sortDesc ? bv - av : av - bv;
      }
      return sortDesc
        ? String(bv).localeCompare(String(av))
        : String(av).localeCompare(String(bv));
    });
    return copy;
  }, [filtered, columns, sortKey, sortDesc]);

  const pageSizeEff = pageSize ?? sorted.length;
  const pageCount = Math.max(1, Math.ceil(sorted.length / pageSizeEff));
  const pageStart = page * pageSizeEff;
  const pageRows = sorted.slice(pageStart, pageStart + pageSizeEff);

  const headerClick = (key: string) => {
    if (key === sortKey) setSortDesc((d) => !d);
    else {
      setSortKey(key);
      const col = columns.find((c) => c.key === key);
      setSortDesc(col?.defaultSortDesc ?? col?.numeric ?? true);
    }
    setPage(0);
  };

  return (
    <div>
      <div className="table-wrap">
        <table className="data">
          <thead>
            <tr>
              {columns.map((c) => (
                <th key={c.key} onClick={() => headerClick(c.key)}
                    className={c.numeric ? "num" : ""}>
                  {c.header}
                  {sortKey === c.key && (sortDesc ? " ▼" : " ▲")}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {pageRows.map((row, i) => (
              <tr key={i} className={rowClassName?.(row)}>
                {columns.map((c) => {
                  const v = c.render ? c.render(row) : c.accessor(row);
                  return <td key={c.key} className={c.numeric ? "num" : ""}>{v as any}</td>;
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {pageSize && pageCount > 1 && (
        <div className="filter-bar" style={{ justifyContent: "flex-end", marginTop: 10 }}>
          <button disabled={page === 0} onClick={() => setPage((p) => p - 1)}>‹ Prev</button>
          <span style={{ fontSize: 12, color: "var(--muted)" }}>
            Page {page + 1} of {pageCount}  ({sorted.length} rows)
          </span>
          <button disabled={page >= pageCount - 1} onClick={() => setPage((p) => p + 1)}>Next ›</button>
        </div>
      )}
    </div>
  );
}
