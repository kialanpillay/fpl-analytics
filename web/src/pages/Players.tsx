import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import {
  createColumnHelper,
  flexRender,
  getCoreRowModel,
  getFilteredRowModel,
  getSortedRowModel,
  useReactTable,
  type SortingState,
} from "@tanstack/react-table";
import { api } from "../lib/api";
import type { Player } from "../lib/types";
import { money, num } from "../lib/format";

const col = createColumnHelper<Player>();

export function Players() {
  const [position, setPosition] = useState("");
  const [sort, setSort] = useState("balanced");
  const [filter, setFilter] = useState("");
  const [onlyUno, setOnlyUno] = useState(false);
  const [onlyValue, setOnlyValue] = useState(false);
  const { data, isPending } = useQuery({
    queryKey: ["players", position, sort],
    queryFn: () => api.players({ position: position || undefined, sort, n: 600 }),
  });

  const rows = useMemo(() => {
    let list = data?.players || [];
    if (onlyUno) list = list.filter((p) => p.unorthodox);
    if (onlyValue) list = list.filter((p) => p.value_flag);
    return list;
  }, [data, onlyUno, onlyValue]);

  const columns = useMemo(
    () => [
      col.accessor("web_name", {
        header: "Player",
        cell: (info) => (
          <Link className="text-accent" to={`/players/${info.row.original.id}`}>
            {info.getValue()}
          </Link>
        ),
      }),
      col.accessor("position", { header: "Pos" }),
      col.accessor("team_short", { header: "Club" }),
      col.accessor("price", { header: "£", cell: (i) => money(i.getValue()) }),
      col.accessor("xp_gw", { header: "xGW", cell: (i) => num(i.getValue()) }),
      col.accessor("xp_horizon", { header: "xH", cell: (i) => num(i.getValue()) }),
      col.accessor("ppp", { header: "PPP", cell: (i) => num(i.getValue()) }),
      col.accessor("consistency", { header: "Cons", cell: (i) => num(i.getValue()) }),
      col.accessor("balanced", { header: "Bal", cell: (i) => num(i.getValue()) }),
      col.accessor("ownership", { header: "Own", cell: (i) => num(i.getValue(), 1) }),
      col.accessor("next_fixture", { header: "Next" }),
    ],
    [],
  );

  const [sorting, setSorting] = useState<SortingState>([]);
  const table = useReactTable({
    data: rows,
    columns,
    state: { sorting, globalFilter: filter },
    onSortingChange: setSorting,
    onGlobalFilterChange: setFilter,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
  });

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <select
          className="rounded border border-line bg-panel px-2 py-1 text-sm"
          value={position}
          onChange={(e) => setPosition(e.target.value)}
        >
          <option value="">All Positions</option>
          {["GKP", "DEF", "MID", "FWD"].map((p) => (
            <option key={p}>{p}</option>
          ))}
        </select>
        <select
          className="rounded border border-line bg-panel px-2 py-1 text-sm"
          value={sort}
          onChange={(e) => setSort(e.target.value)}
        >
          {(
            [
              ["balanced", "Balanced"],
              ["aggressive", "Aggressive"],
              ["template", "Template"],
              ["xp_horizon", "Horizon xPts"],
              ["xp_gw", "GW xPts"],
              ["ppp", "PPP"],
              ["consistency", "Consistency"],
              ["residual", "Residual"],
              ["differential", "Differential"],
            ] as const
          ).map(([value, label]) => (
            <option key={value} value={value}>
              {label}
            </option>
          ))}
        </select>
        <input
          className="rounded border border-line bg-panel px-2 py-1 text-sm"
          placeholder="Filter…"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
        />
        <label className="text-xs text-mute">
          <input type="checkbox" className="mr-1" checked={onlyUno} onChange={(e) => setOnlyUno(e.target.checked)} />
          Unorthodox
        </label>
        <label className="text-xs text-mute">
          <input type="checkbox" className="mr-1" checked={onlyValue} onChange={(e) => setOnlyValue(e.target.checked)} />
          Underpriced
        </label>
      </div>
      {isPending ? (
        <p className="text-mute">Loading Pool…</p>
      ) : (
        <div className="card overflow-x-auto rounded-xl p-3">
          <table className="w-full min-w-[880px] text-left text-sm">
            <thead className="text-xs text-mute">
              {table.getHeaderGroups().map((hg) => (
                <tr key={hg.id}>
                  {hg.headers.map((h) => (
                    <th
                      key={h.id}
                      className="cursor-pointer pb-2 pr-3 font-medium"
                      onClick={h.column.getToggleSortingHandler()}
                    >
                      {flexRender(h.column.columnDef.header, h.getContext())}
                    </th>
                  ))}
                </tr>
              ))}
            </thead>
            <tbody>
              {table.getRowModel().rows.map((row) => (
                <tr key={row.id} className="border-t border-line">
                  {row.getVisibleCells().map((cell) => (
                    <td key={cell.id} className="py-1.5 pr-3">
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
