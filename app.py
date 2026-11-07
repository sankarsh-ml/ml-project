"""TESS Planet Explorer: python app.py"""
import argparse
import json
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import joblib
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from tess_pipeline import ROOT, CONFIG, analyze, download_tic
from classifier import predict, MODEL_PATH
from search_pipeline import detection_accepted
from target_selector import TargetIndex, SearchableTIC, parse_tic, EXAMPLES

BG, PANEL, TEXT, MUTED, ACCENT = "#0b1220", "#142033", "#e7edf7", "#9cafc9", "#55dbc4"


class Explorer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TESS Planet Explorer")
        self.geometry("1230x850")
        self.minsize(1050, 760)
        self.configure(bg=BG)
        self.events = queue.Queue()
        self.result = None
        self.busy = False
        self.model_path = MODEL_PATH
        self.report = {}
        if (ROOT / "training_report.json").exists():
            self.report = json.loads((ROOT / "training_report.json").read_text())
        self.target_index = TargetIndex()
        self.example_cursor = 0
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TButton", font=("Segoe UI", 10), padding=6,
                        background="#243b55", foreground=TEXT)
        style.map("TButton", background=[("active", "#345875")])
        style.configure("TEntry", fieldbackground="#23334b", foreground=TEXT, padding=7)
        style.configure("TCombobox", fieldbackground="#23334b", foreground=TEXT, padding=6)
        self.option_add("*TCombobox*Listbox.background", "#23334b")
        self.option_add("*TCombobox*Listbox.foreground", TEXT)
        style.configure("TProgressbar", background=ACCENT, troughcolor=PANEL)

        header = tk.Frame(self, bg=BG)
        header.pack(fill="x", padx=24, pady=(18, 12))
        tk.Label(header, text="TESS / PLANET EXPLORER", bg=BG, fg=ACCENT,
                 font=("Segoe UI", 12, "bold")).pack(anchor="w")
        tk.Label(header, text="Find the signal in starlight.", bg=BG, fg=TEXT,
                 font=("Segoe UI", 25, "bold")).pack(anchor="w")
        tk.Label(header, text="Choose a star to look for repeating dips in its brightness.",
                 bg=BG, fg=MUTED, font=("Segoe UI", 10)).pack(anchor="w", pady=(4, 0))

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=24)
        side = tk.Frame(body, bg=PANEL, width=265, padx=18, pady=14)
        side.pack(side="left", fill="y", padx=(0, 16))
        side.pack_propagate(False)
        self.label(side, "OBSERVATION", ACCENT, 11, True)
        self.label(side, "TIC ID", MUTED)
        self.tic = tk.StringVar(value=EXAMPLES[0])
        self.tic_selector = SearchableTIC(side, self.tic, self.target_index, self.on_tic_selected)
        self.tic_selector.pack(fill="x", pady=(4, 5))
        self.target_hint = tk.StringVar(value="Type a TIC ID or browse saved targets.")
        tk.Label(side, textvariable=self.target_hint, wraplength=225, justify="left", anchor="w",
                 bg=PANEL, fg=MUTED, font=("Segoe UI", 9)).pack(fill="x", pady=(0, 10))
        self.label(side, "TESS sector", MUTED)
        self.sector = tk.StringVar(value="1")
        self.sector_selector = ttk.Combobox(side, textvariable=self.sector, state="normal")
        self.sector_selector.pack(fill="x", pady=(4, 14))
        self.on_tic_selected(self.tic.get())
        self.analyze_button = ttk.Button(side, text="Analyze TIC  →", command=self.run_tic)
        self.analyze_button.pack(fill="x", pady=(0, 8))
        self.file_button = ttk.Button(side, text="Open local FITS…", command=self.run_file)
        self.file_button.pack(fill="x", pady=(0, 8))
        self.example_button = ttk.Button(side, text="Load example target", command=self.test_star)
        self.example_button.pack(fill="x", pady=(0, 8))
        self.export_button = ttk.Button(side, text="Export result…", command=self.export, state="disabled")
        self.export_button.pack(fill="x", pady=(0, 10))
        self.details_button = ttk.Button(side, text="Measurement details", command=self.show_details, state="disabled")
        self.details_button.pack(fill="x", pady=(0, 8))
        ttk.Button(side, text="About the model", command=self.evaluation).pack(fill="x", pady=(0, 12))
        self.label(side, "Experimental prediction.\nFollow-up is needed to\nconfirm a planet.", MUTED, 9)

        main = tk.Frame(body, bg=BG)
        main.pack(side="left", fill="both", expand=True)
        self.verdict = tk.StringVar(value="Ready to analyze a star")
        tk.Label(main, textvariable=self.verdict, bg=BG, fg=ACCENT,
                 font=("Segoe UI", 18, "bold"), anchor="w").pack(fill="x")
        self.detail = tk.StringVar(value="Download an observation or open a local TESS SPOC light curve.")
        tk.Label(main, textvariable=self.detail, bg=BG, fg=MUTED, justify="left",
                 anchor="w", wraplength=850, font=("Segoe UI", 10)).pack(fill="x", pady=(5, 12))
        cards = tk.Frame(main, bg=BG)
        cards.pack(fill="x", pady=(0, 10))
        self.values = []
        for title, hint in [("ORBITAL PERIOD", "Time between repeating dips"),
                            ("TRANSIT DURATION", "How long each dip lasts"),
                            ("BRIGHTNESS DROP", "Light blocked during the dip")]:
            card = tk.Frame(cards, bg=PANEL, padx=12, pady=10)
            card.pack(side="left", fill="x", expand=True, padx=(0, 6))
            tk.Label(card, text=title, bg=PANEL, fg=MUTED, font=("Segoe UI", 8)).pack(anchor="w")
            variable = tk.StringVar(value="—")
            self.values.append(variable)
            tk.Label(card, textvariable=variable, bg=PANEL, fg=TEXT,
                     font=("Segoe UI", 18, "bold")).pack(anchor="w")
            tk.Label(card, text=hint, bg=PANEL, fg=MUTED,
                     font=("Segoe UI", 8)).pack(anchor="w", pady=(3, 0))

        self.figure = Figure(figsize=(8.5, 5.7), facecolor=BG)
        self.axes = self.figure.subplots(3, 1)
        self.figure.subplots_adjust(left=0.085, right=0.98, top=0.91, bottom=0.1, hspace=0.9)
        self.style_axes()
        self.canvas = FigureCanvasTkAgg(self.figure, master=main)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.draw()

        footer = tk.Frame(self, bg=BG)
        footer.pack(side="bottom", fill="x", padx=24, pady=12, before=body)
        self.progress = ttk.Progressbar(footer, mode="indeterminate", length=150)
        self.progress.pack(side="right")
        self.status = tk.StringVar(value="Ready. Choose a star and select Analyze TIC.")
        tk.Label(footer, textvariable=self.status, bg=BG, fg=MUTED,
                 font=("Segoe UI", 10), anchor="w").pack(side="left", fill="x")
        self.after(100, self.poll)

    def label(self, parent, text, color=TEXT, size=10, bold=False):
        label = tk.Label(parent, text=text, bg=PANEL, fg=color, justify="left",
                         anchor="w", font=("Segoe UI", size, "bold" if bold else "normal"))
        label.pack(fill="x", pady=(0, 6))
        return label

    def style_axes(self):
        for ax, title, xlabel, ylabel in zip(self.axes,
                ["Brightness over time - highlighted dips", "Repeating dips aligned", "Transit period search (BLS)"],
                ["Days since observation started", "Hours from the middle of a dip", "Repeating period (days)"],
                ["Brightness (%)", "Change (%)", "Signal strength"]):
            ax.set_facecolor(PANEL)
            ax.set_title(title, color=TEXT, fontsize=10, loc="left", pad=8)
            ax.set_xlabel(xlabel, color=MUTED, fontsize=8)
            ax.set_ylabel(ylabel, color=MUTED, fontsize=8)
            ax.tick_params(colors=MUTED, labelsize=8)
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            for spine in ax.spines.values():
                spine.set_color("#34455e")
            ax.grid(alpha=0.08, color="white")

    def on_tic_selected(self, value):
        try:
            tic = parse_tic(value)
        except ValueError:
            return
        record = self.target_index.get(tic)
        sectors = sorted(record["sectors"]) if record else []
        self.sector_selector.configure(values=[str(sector) for sector in sectors])
        if sectors and self.sector.get() not in list(map(str, sectors)):
            self.sector.set(str(sectors[0]))
        if record:
            from target_selector import STATUS
            statuses = ", ".join(sorted({STATUS.get(code, code).split(" (")[0] for code in record["statuses"]})) or "Downloaded target"
            self.target_hint.set("Catalog: "+statuses)

        else:
            self.target_hint.set("New TIC ID — choose a sector to search.")

    def test_star(self):
        tic = EXAMPLES[self.example_cursor % len(EXAMPLES)]
        self.example_cursor += 1
        self.tic.set(tic)
        self.on_tic_selected(tic)
        self.run_tic()

    def run_tic(self):
        try:
            tic = int(parse_tic(self.tic.get()))
            sector = int(self.sector.get())
            if tic <= 0 or sector <= 0:
                raise ValueError()
        except ValueError:
            messagebox.showerror("Invalid input", "Enter positive numeric TIC and sector values.")
            return
        self.tic_selector.hide()
        self.start(lambda: download_tic(tic, sector))

    def run_file(self):
        path = filedialog.askopenfilename(filetypes=[("TESS FITS", "*.fits"), ("All files", "*.*")])
        if path:
            self.start(lambda: Path(path))

    def start(self, source):
        if self.busy:
            return
        if not self.model_path.exists():
            messagebox.showerror("Model missing", "Run python train_real_tess.py --stage fit first.")
            return
        self.busy = True
        for button in [self.analyze_button, self.file_button, self.example_button, self.export_button, self.details_button]:
            button.configure(state="disabled")
        self.result = None
        self.verdict.set("Analyzing observation…")
        self.detail.set("Downloading and processing a TESS observation. The interface remains available while the search runs.")
        for variable in self.values:
            variable.set("—")
        for ax in self.axes:
            ax.clear()
        self.style_axes()
        self.canvas.draw_idle()
        self.status.set("Locating / downloading TESS light curve…")
        self.progress.start(12)

        def work():
            try:
                artifact = joblib.load(self.model_path)
                if artifact.get("pipeline") != CONFIG:
                    raise ValueError("Model preprocessing settings differ. Retrain with this pipeline.")
                result = analyze(source(), lambda msg: self.events.put(("status", msg)))
                result["threshold"] = artifact["threshold"]
                if detection_accepted(result["features"]):
                    result["prediction"] = predict(pd.DataFrame([result["features"]]), self.model_path)
                else:
                    result["prediction"] = pd.DataFrame([{**result["features"],
                        "planet_candidate_probability": np.nan, "predicted_label": -1, "probability_reliable": False}])
                result["prediction"]["probability_reliable"] &= self.report.get("probability_reliable", False)
                for name, value in result.get("uncertainty", {}).get("std", {}).items():
                    result["prediction"][name+"_bootstrap_std"] = value
                outside = sum(
                    result["features"][name] < minimum or result["features"][name] > artifact["feature_max"][name]
                    for name, minimum in artifact.get("feature_min", {}).items())
                result["outside_training_range"] = outside > len(artifact["features"])*.2
                self.events.put(("done", result))
            except Exception as error:
                self.events.put(("error", str(error)))
        threading.Thread(target=work, daemon=True).start()

    def poll(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "status":
                    self.status.set(value)
                else:
                    self.busy = False
                    self.progress.stop()
                    for button in [self.analyze_button, self.file_button, self.example_button]:
                        button.configure(state="normal")
                    if kind == "error":
                        self.verdict.set("Observation could not be classified")
                        self.detail.set(value)
                        self.status.set("Check the TIC, sector, connection, or FITS file and retry.")
                    else:
                        self.result = value
                        self.render(value)
                        self.export_button.configure(state="normal")
                        self.details_button.configure(state="normal")
        except queue.Empty:
            pass
        self.after(100, self.poll)

    def render(self, result):
        f = result["features"]
        row = result["prediction"].iloc[0]
        probability = row.planet_candidate_probability
        if row.predicted_label == -1:
            self.verdict.set("Insufficient periodic signal")
        else:
            self.verdict.set("Planet-like candidate" if row.predicted_label else "False-positive-like signal")
        self.target_index.add_observation(f["tic_id"], f["sector"])
        description = f"TIC {f['tic_id']} | Sector {f['sector']} | {f['observed_transits']} dips checked"
        notices = []
        threshold = result.get("threshold", .5)
        if np.isfinite(probability) and abs(probability-threshold) < .05:
            notices.append("Borderline prediction; review with care.")
        record = self.target_index.get(f["tic_id"])
        statuses = record["statuses"] if record else set()
        if (row.predicted_label == 0 and statuses & {"CP", "KP"}) or (row.predicted_label == 1 and "FP" in statuses):
            notices.append("This prediction disagrees with the catalog.")
        if result.get("outside_training_range"):
            notices.append("Unusual signal for this model; review with care.")
        if notices:
            description += "\n"+" ".join(notices)
        self.detail.set(description)
        for variable, value in zip(self.values, [f"{f['period_days']:.4f} days",
                f"{f['duration_days']*24:.2f} hours", f"{f['depth']*100:.4f}%"]):
            variable.set(value)
        for ax in self.axes:
            ax.clear()
        self.style_axes()
        t, flux, phase = result["time"], result["flux"], result["phase"]
        in_transit = np.abs(phase) < f["duration_days"] / 2
        self.axes[0].scatter(t-t[0], flux*100, s=1, color="#779ac7", alpha=0.55, rasterized=True)
        self.axes[0].scatter(t[in_transit]-t[0], flux[in_transit]*100, s=3, color=ACCENT)
        limit = min(f["period_days"]/2, max(0.25, f["duration_days"]*2.5))
        visible = np.abs(phase) <= limit
        self.axes[1].scatter(phase[visible]*24, (flux[visible]-1)*100, s=2, color="#779ac7", alpha=0.18)
        edges = np.linspace(-limit, limit, 65)
        mids, medians = [], []
        for left, right in zip(edges[:-1], edges[1:]):
            selected = (phase >= left) & (phase < right)
            if selected.any():
                mids.append((left+right)*12)
                medians.append((np.median(flux[selected])-1)*100)
        self.axes[1].plot(mids, medians, ".-", color=ACCENT, markersize=4, linewidth=1)
        self.axes[1].set_xlim(-limit*24, limit*24)
        self.axes[2].plot(result["period_grid"], result["power"], color=ACCENT, linewidth=0.8)
        self.axes[2].axvline(f["period_days"], color="#ffca76", alpha=0.8)
        self.canvas.draw_idle()
        self.status.set("Analysis complete. Export result saves the measurements and plots.")

    def measurement_details(self):
        if not self.result:
            return "Analyze a star to see its measurements."
        f = self.result["features"]
        membership = "Not used to train this model."
        if f["tic_id"] in self.report.get("train_tic_ids", []):
            membership = "Used in training; this result is not an independent test."
        elif f["tic_id"] in self.report.get("test_tic_ids", []):
            membership = "Part of the independent evaluation set."
        lines = [f"TIC {f['tic_id']} / Sector {f['sector']}", self.verdict.get(), "",
            f"Orbital period: {f['period_days']:.6f} days",
            f"Transit duration: {f['duration_days']*24:.4f} hours",
            f"Brightness drop: {f['depth']*100:.4f}% ({f['depth']*1e6:.1f} ppm)",
            f"Sampled transit windows: {f['observed_transits']}", "", membership]
        uncertainty = self.result.get("uncertainty", {})
        if uncertainty:
            std = uncertainty["std"]
            def scatter(value, unit):
                return f"{value:.6g} {unit}" if value > 0 else "No variation across resamples (grid-limited)."
            lines += ["", f"Measurement uncertainty ({uncertainty['resamples']} resampled fits):",
                "Period scatter: "+scatter(std["period_days"], "days"),
                "Duration scatter: "+scatter(std["duration_days"]*24, "hours"),
                "Depth scatter: "+scatter(std["depth"]*1e6, "ppm"),
                "These local estimates assume the selected period and fixed detrending."]
        return "\n".join(lines)

    def show_details(self):
        messagebox.showinfo("Measurement details", self.measurement_details())

    def export(self):
        if not self.result:
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="tess_prediction.csv",
                                          filetypes=[("CSV", "*.csv")])
        if path:
            self.save_result(path)

    def save_result(self, path):
        self.result["prediction"].to_csv(path, index=False)
        self.figure.savefig(str(Path(path).with_suffix(".png")), dpi=160, facecolor=BG)
        self.status.set("Saved prediction CSV and matching plot PNG.")

    def evaluation(self):
        if not self.report:
            messagebox.showinfo("Evaluation", "No training report found yet.")
            return
        messagebox.showinfo("About the model", (
            "BLS finds repeating brightness dips.\n"
            "A Random Forest compares their measurements with labeled TESS signals.\n\n"
            "Planet-like candidate: resembles the planet examples.\n"
            "False-positive-like signal: resembles the false-positive examples.\n"
            "Insufficient periodic signal: too weak for this classifier.\n\n"
            f"Independent accuracy: {self.report['accuracy']:.2%}\n"
            f"Balanced accuracy: {self.report['balanced_accuracy']:.2%}\n"
            f"Training stars: {self.report['train_rows']} | Evaluated signals: {self.report['test_rows']}\n\n"
            "The model can disagree with known catalog labels. It does not confirm a planet.\n"
            "See README.md for the full method and evaluation limits."))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="Analyze a cached example target on startup")
    args = parser.parse_args()
    app = Explorer()
    if args.demo:
        app.after(800, app.test_star)
    app.mainloop()
