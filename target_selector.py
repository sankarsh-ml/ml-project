"""Searchable editable TIC selector and local catalog/inventory index."""
from pathlib import Path
import re
import tkinter as tk
from tkinter import ttk
import pandas as pd

ROOT = Path(__file__).resolve().parent
STATUS = {"CP": "Known planet (CP)", "KP": "Known planet (KP)", "PC": "Candidate (PC)",
          "FP": "False positive (FP)", "FA": "False alarm (FA)", "APC": "Ambiguous (APC)"}
EXAMPLES = ["149603524", "231670397", "183979262", "261136679"]


def parse_tic(value):
    text = str(value).strip()
    match = re.fullmatch(r"(?:TIC\s*)?(\d+)", text, flags=re.I)
    if not match or int(match[1]) <= 0:
        raise ValueError("Enter a positive TIC ID, for example TIC 149603524.")
    return str(int(match[1]))


class TargetIndex:
    def __init__(self, root=ROOT):
        self.root, self.records = Path(root), {}
        data = self.root / "data"
        catalog = data / "toi_catalog_full.csv"
        if not catalog.exists():
            catalog = data / "toi_catalog.csv"
        if catalog.exists():
            frame = pd.read_csv(catalog, dtype={"tid": str})
            for row in frame.itertuples():
                try:
                    tic = parse_tic(row.tid)
                except ValueError:
                    continue
                record = self.records.setdefault(tic, {"tic_id": tic, "statuses": set(), "sectors": set()})
                record["statuses"].add(str(row.tfopwg_disp))
        sources = [(data / "sector_index.csv", "tic_id"),
                   (data / "v4/available_observations.csv", "tic_id"),
                   (data / "training_candidates.csv", "tic_id"),
                   (data / "v4/observation_plan.csv", "tic_id")]
        sources.append((data / "v5/observation_plan.csv", "tic_id"))
        for path, column in sources:
            if path.exists():
                frame = pd.read_csv(path, usecols=lambda col: col in [column, "sector"], dtype={column: str})
                if "sector" not in frame:
                    continue
                for tic, sector in frame[[column, "sector"]].dropna().itertuples(index=False, name=None):
                    self.add_observation(tic, int(sector))
        for path in (data / "lightcurves").glob("*.fits"):
            match = re.search(r"-s(\d+)-(\d{16})-", path.name)
            if match:
                self.add_observation(match[2], int(match[1]))

    def add_observation(self, tic, sector):
        tic = parse_tic(tic)
        record = self.records.setdefault(tic, {"tic_id": tic, "statuses": set(), "sectors": set()})
        record["sectors"].add(int(sector))

    def get(self, tic):
        return self.records.get(parse_tic(tic))

    def label(self, record):
        statuses = ", ".join(STATUS.get(code, code) for code in sorted(record["statuses"])) or "Downloaded observation"
        sectors = ",".join(map(str, sorted(record["sectors"])))
        if len(sectors) > 50:
            sectors = sectors[:47]+"…"
        return f"TIC {record['tic_id']}  ·  {statuses}  ·  Sectors {sectors or 'unknown'}"

    def search(self, query, limit=80):
        query = query.strip().casefold().removeprefix("tic").strip()
        matches = []
        for tic, record in self.records.items():
            if not query or query in tic or query in self.label(record).casefold():
                matches.append(record)
        matches.sort(key=lambda row: (not (query and row["tic_id"].startswith(query)),
            EXAMPLES.index(row["tic_id"]) if row["tic_id"] in EXAMPLES else len(EXAMPLES), int(row["tic_id"])))
        return matches[:limit]


class SearchableTIC(tk.Frame):
    """Entry keeps focus during filtering; popup supports arrows and Enter."""
    def __init__(self, master, variable, index, on_select, bg="#142033", **kwargs):
        super().__init__(master, bg=bg, **kwargs)
        self.variable, self.index, self.on_select = variable, index, on_select
        self.rows, self.active, self.pending = [], -1, None
        self.suppress = False
        self.entry = tk.Entry(self, textvariable=variable, bg="#23334b", fg="#e7edf7", insertbackground="#e7edf7",
            relief="flat", font=("Segoe UI", 11), highlightthickness=1, highlightbackground="#40536b", highlightcolor="#55dbc4")
        self.entry.pack(side="left", fill="both", expand=True, ipady=7)
        self.arrow = tk.Button(self, text="▾", command=self.toggle, bg="#23334b", fg="#55dbc4",
            activebackground="#304866", activeforeground="white", relief="flat", width=3, takefocus=False)
        self.arrow.pack(side="right", fill="y")
        self.popup = tk.Toplevel(self)
        self.popup.withdraw()
        self.popup.overrideredirect(True)
        self.popup.transient(self.winfo_toplevel())
        self.popup.configure(bg="#40536b")
        border = tk.Frame(self.popup, bg="#142033", padx=1, pady=1)
        border.pack(fill="both", expand=True, padx=1, pady=1)
        self.listbox = tk.Listbox(border, bg="#142033", fg="#e7edf7", selectbackground="#285368",
            selectforeground="#ffffff", activestyle="none", highlightthickness=0, relief="flat",
            font=("Segoe UI", 10), exportselection=False, height=8)
        scroll = ttk.Scrollbar(border, orient="vertical", command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.listbox.pack(fill="both", expand=True)
        self.entry.bind("<Down>", lambda event: self.move(1))
        self.entry.bind("<Up>", lambda event: self.move(-1))
        self.entry.bind("<Return>", self.accept)
        self.entry.bind("<Escape>", lambda event: self.hide())
        self.entry.bind("<FocusOut>", lambda event: self.after(180, self.hide_if_unfocused))
        self.listbox.bind("<ButtonRelease-1>", self.click_selection)
        self.listbox.bind("<Return>", self.accept)
        self.listbox.bind("<Escape>", lambda event: self.hide())
        self.variable.trace_add("write", self.changed)
        self.root_window = self.winfo_toplevel()
        self.configure_binding = self.root_window.bind("<Configure>", self.root_changed, add=True)

    def root_changed(self, event):
        if event.widget == self.root_window and self.winfo_exists():
            self.hide()

    def changed(self, *args):
        if self.suppress:
            return
        self.active = -1
        if self.pending:
            self.after_cancel(self.pending)
        self.pending = self.after(60, self.filter_rows)

    def filter_rows(self, show=True, query=None):
        if self.pending:
            self.after_cancel(self.pending)
        self.pending = None
        self.rows = self.index.search(self.variable.get() if query is None else query)
        self.listbox.delete(0, "end")
        for row in self.rows:
            self.listbox.insert("end", self.index.label(row))
        if not self.rows:
            self.listbox.insert("end", "No saved match — type any TIC ID and press Enter.")
        if show and self.focus_get() == self.entry:
            self.show()

    def show(self):
        self.update_idletasks()
        width = min(760, max(570, self.winfo_width()))
        x = min(self.winfo_rootx(), self.winfo_screenwidth()-width-12)
        height = min(8, max(1, len(self.rows)))*25+8
        y = self.winfo_rooty()+self.winfo_height()+3
        if y+height > self.winfo_screenheight():
            y = self.winfo_rooty()-height-3
        self.popup.geometry(f"{width}x{height}+{max(0,x)}+{max(0,y)}")
        self.popup.deiconify()
        self.popup.lift()

    def hide(self):
        if self.pending:
            self.after_cancel(self.pending)
            self.pending = None
        if self.popup.winfo_exists():
            self.popup.withdraw()
        return "break"

    def destroy(self):
        if self.pending:
            self.after_cancel(self.pending)
            self.pending = None
        if self.configure_binding:
            self.root_window.unbind("<Configure>", self.configure_binding)
            self.configure_binding = None
        if self.popup.winfo_exists():
            self.popup.destroy()
        super().destroy()

    def hide_if_unfocused(self):
        focused = self.focus_get()
        if focused not in [self.entry, self.listbox]:
            self.hide()

    def toggle(self):
        if self.popup.winfo_viewable():
            self.hide()
        else:
            self.entry.focus_set()
            self.filter_rows(query="")

    def move(self, direction):
        if self.pending or not self.popup.winfo_viewable():
            self.filter_rows()
        if self.rows:
            self.active = min(max(self.active+direction, 0), len(self.rows)-1)
            self.listbox.selection_clear(0, "end")
            self.listbox.selection_set(self.active)
            self.listbox.activate(self.active)
            self.listbox.see(self.active)
        return "break"

    def click_selection(self, event):
        self.active = self.listbox.nearest(event.y)
        return self.accept()

    def accept(self, event=None):
        if self.popup.winfo_viewable() and self.rows and self.active >= 0:
            tic = self.rows[self.active]["tic_id"]
        else:
            try:
                tic = parse_tic(self.variable.get())
            except ValueError:
                return "break"
        self.suppress = True
        self.variable.set(tic)
        self.suppress = False
        if self.pending:
            self.after_cancel(self.pending)
            self.pending = None
        self.hide()
        self.on_select(tic)
        return "break"


def build_sector_index():
    """Index all catalog TICs against already-downloaded official manifests."""
    catalog = pd.read_csv(ROOT / "data/toi_catalog_full.csv", usecols=["tid"])
    ids = set(catalog.tid.astype(int))
    rows = set()
    pattern = re.compile(r"tess\d+-s(\d+)-(\d{16})-\d+-s_lc\.fits")
    for path in (ROOT / "data").glob("sector*_download.sh"):
        for sector, digits in pattern.findall(path.read_text()):
            tic = int(digits)
            if tic in ids:
                rows.add((str(tic), int(sector)))
    pd.DataFrame(sorted(rows), columns=["tic_id", "sector"]).to_csv(ROOT / "data/sector_index.csv", index=False)
    print(f"Indexed {len(rows)} catalog TIC/sector pairs.")


if __name__ == "__main__":
    build_sector_index()
