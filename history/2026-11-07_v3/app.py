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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from tess_pipeline import ROOT, CONFIG, analyze, download_tic
from svm_classifier import predict

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
        self.model_path = ROOT / "svm_model.joblib"
        self.report = {}
        if (ROOT / "training_report.json").exists():
            self.report = json.loads((ROOT / "training_report.json").read_text())
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TButton", font=("Segoe UI", 10), padding=6,
                        background="#243b55", foreground=TEXT)
        style.map("TButton", background=[("active", "#345875")])
        style.configure("TEntry", fieldbackground="#23334b", foreground=TEXT, padding=7)
        style.configure("TProgressbar", background=ACCENT, troughcolor=PANEL)

        header = tk.Frame(self, bg=BG)
        header.pack(fill="x", padx=24, pady=(18, 12))
        tk.Label(header, text="TESS / PLANET EXPLORER", bg=BG, fg=ACCENT,
                 font=("Segoe UI", 12, "bold")).pack(anchor="w")
        tk.Label(header, text="Find the signal in starlight.", bg=BG, fg=TEXT,
                 font=("Segoe UI", 25, "bold")).pack(anchor="w")
        tk.Label(header, text="On-demand archive analysis  /  Two-minute TESS light curves  /  BLS + SVM",
                 bg=BG, fg=MUTED, font=("Segoe UI", 10)).pack(anchor="w", pady=(4, 0))

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=24)
        side = tk.Frame(body, bg=PANEL, width=265, padx=18, pady=14)
        side.pack(side="left", fill="y", padx=(0, 16))
        side.pack_propagate(False)
        self.label(side, "OBSERVATION", ACCENT, 11, True)
        self.label(side, "TIC ID", MUTED)
        demo = next((row for row in self.report.get("test_predictions", []) if row["label"] == 1), {})
        self.tic = tk.StringVar(value=demo.get("tic_id", "261136679"))
        ttk.Entry(side, textvariable=self.tic).pack(fill="x", pady=(4, 14))
        self.label(side, "TESS sector", MUTED)
        self.sector = tk.StringVar(value=str(demo.get("sector", self.report.get("sector", 1))))
        ttk.Entry(side, textvariable=self.sector).pack(fill="x", pady=(4, 14))
        self.analyze_button = ttk.Button(side, text="Analyze TIC  →", command=self.run_tic)
        self.analyze_button.pack(fill="x", pady=(0, 8))
        self.file_button = ttk.Button(side, text="Open local FITS…", command=self.run_file)
        self.file_button.pack(fill="x", pady=(0, 8))
        self.example_button = ttk.Button(side, text="Use held-out test star", command=self.test_star)
        self.example_button.pack(fill="x", pady=(0, 8))
        self.export_button = ttk.Button(side, text="Export result…", command=self.export, state="disabled")
        self.export_button.pack(fill="x", pady=(0, 10))
        self.label(side, "MODEL / REAL DATA", ACCENT, 11, True)
        if self.report:
            metrics = self.report["metrics"]
            sectors = ", ".join(map(str, self.report.get("sectors", [self.report["sector"]])))
            summary = (f"Sectors {sectors} pilot\n{self.report['matched_rows']} observations / "
                       f"{self.report.get('unique_stars', self.report['matched_rows'])} stars\n"
                       f"{len(self.report['test_tic_ids'])} development test stars\n\n"
                       f"Accuracy   {metrics['accuracy']:.1%}\n"
                       f"Balanced   {self.report.get('balanced_accuracy', 0):.1%}\n"
                       f"Planet F1   {metrics['1']['f1-score']:.3f}")
        else:
            summary = "Model not trained yet.\nRun train_real_tess.py first."
        self.label(side, summary, TEXT, 10)
        ttk.Button(side, text="View evaluation", command=self.evaluation).pack(fill="x", pady=8)
        self.label(side, "Research prototype\nScores do not confirm a planet.\nSee evaluation for limitations.", MUTED, 9)

        main = tk.Frame(body, bg=BG)
        main.pack(side="left", fill="both", expand=True)
        self.verdict = tk.StringVar(value="Ready to analyze a star")
        if self.report and self.report.get("balanced_accuracy", 0) < 0.6:
            self.verdict.set("Experimental model · validation is weak")
        tk.Label(main, textvariable=self.verdict, bg=BG, fg=ACCENT,
                 font=("Segoe UI", 18, "bold"), anchor="w").pack(fill="x")
        self.detail = tk.StringVar(value="Download an observation or open a local TESS SPOC light curve.")
        tk.Label(main, textvariable=self.detail, bg=BG, fg=MUTED, justify="left",
                 anchor="w", wraplength=850, font=("Segoe UI", 10)).pack(fill="x", pady=(5, 12))
        cards = tk.Frame(main, bg=BG)
        cards.pack(fill="x", pady=(0, 10))
        self.values = []
        for title in ["PERIOD / DAYS", "DURATION / HOURS", "DEPTH / PPM", "CANDIDATE SCORE"]:
            card = tk.Frame(cards, bg=PANEL, padx=12, pady=10)
            card.pack(side="left", fill="x", expand=True, padx=(0, 6))
            tk.Label(card, text=title, bg=PANEL, fg=MUTED, font=("Segoe UI", 8)).pack(anchor="w")
            variable = tk.StringVar(value="—")
            self.values.append(variable)
            tk.Label(card, textvariable=variable, bg=PANEL, fg=TEXT,
                     font=("Segoe UI", 18, "bold")).pack(anchor="w")

        self.figure = Figure(figsize=(8.5, 5.7), facecolor=BG)
        self.axes = self.figure.subplots(3, 1)
        self.figure.subplots_adjust(left=0.085, right=0.98, top=0.91, bottom=0.1, hspace=0.9)
        self.style_axes()
        self.canvas = FigureCanvasTkAgg(self.figure, master=main)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, main, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x", before=self.canvas.get_tk_widget())
        self.canvas.draw()

        footer = tk.Frame(self, bg=BG)
        footer.pack(side="bottom", fill="x", padx=24, pady=12, before=body)
        self.progress = ttk.Progressbar(footer, mode="indeterminate", length=150)
        self.progress.pack(side="right")
        self.status = tk.StringVar(value="Ready. Internet is required only for uncached downloads.")
        tk.Label(footer, textvariable=self.status, bg=BG, fg=MUTED,
                 font=("Segoe UI", 10), anchor="w").pack(side="left", fill="x")
        self.after(100, self.poll)

    def label(self, parent, text, color=TEXT, size=10, bold=False):
        label = tk.Label(parent, text=text, bg=PANEL, fg=color, justify="left",
                         anchor="w", font=("Segoe UI", size, "bold" if bold else "normal"))
        label.pack(fill="x", pady=(0, 6))
        return label

    def style_axes(self):
        for ax, title, xlabel in zip(self.axes,
                ["Flattened starlight · highlighted transit windows", "Folded transit · binned profile", "BLS period search"],
                ["Time / BTJD (days)", "Time from transit center / hours", "Trial period / days"]):
            ax.set_facecolor(PANEL)
            ax.set_title(title, color=TEXT, fontsize=10, loc="left", pad=8)
            ax.set_xlabel(xlabel, color=MUTED, fontsize=8)
            ax.tick_params(colors=MUTED, labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#34455e")
            ax.grid(alpha=0.08, color="white")

    def test_star(self):
        predictions = self.report.get("test_predictions", [])
        if not predictions:
            messagebox.showinfo("No test data", "Train the real-data model first.")
            return
        # Prefer a held-out positive for demonstrating the transit plot.
        row = next((p for p in predictions if p["label"] == 1), predictions[0])
        self.tic.set(row["tic_id"])
        self.sector.set(str(row.get("sector", self.report["sector"])))
        self.run_tic()

    def run_tic(self):
        try:
            tic = int(self.tic.get().upper().replace("TIC", "").strip())
            sector = int(self.sector.get())
            if tic <= 0 or sector <= 0:
                raise ValueError()
        except ValueError:
            messagebox.showerror("Invalid input", "Enter positive numeric TIC and sector values.")
            return
        self.start(lambda: download_tic(tic, sector))

    def run_file(self):
        path = filedialog.askopenfilename(filetypes=[("TESS FITS", "*.fits"), ("All files", "*.*")])
        if path:
            self.start(lambda: Path(path))

    def start(self, source):
        if self.busy:
            return
        if not self.model_path.exists():
            messagebox.showerror("Model missing", "Run python train_real_tess.py --sectors 1 2 3 first.")
            return
        self.busy = True
        for button in [self.analyze_button, self.file_button, self.example_button, self.export_button]:
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
                result["prediction"] = predict(pd.DataFrame([result["features"]]), self.model_path)
                result["outside_training_range"] = any(
                    result["features"][name] < minimum or
                    result["features"][name] > artifact["feature_max"][name]
                    for name, minimum in artifact.get("feature_min", {}).items())
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
        except queue.Empty:
            pass
        self.after(100, self.poll)

    def render(self, result):
        f = result["features"]
        row = result["prediction"].iloc[0]
        probability = row.planet_candidate_probability
        self.verdict.set("Planet-like · experimental" if row.predicted_label else "False-positive-like · experimental")
        membership = "New star relative to this model"
        if f["tic_id"] in self.report.get("train_tic_ids", []):
            membership = "Training star — this result is not an independent test"
        elif f["tic_id"] in self.report.get("test_tic_ids", []):
            membership = "Held-out evaluation star"
        threshold = self.report.get("threshold", 0.5)
        uncertain = " · Near decision threshold" if abs(probability - threshold) < 0.05 else ""
        if result.get("outside_training_range"):
            uncertain += " · Outside pilot feature range"
        validation = ""
        if self.report.get("balanced_accuracy", 0) < 0.6:
            validation = f"Weak validation: {self.report['metrics']['accuracy']:.0%} development accuracy. This result is not reliable.\n"
        self.detail.set(validation + f"TIC {f['tic_id']} · Sector {f['sector']} · {f['observed_transits']} sampled transit windows\n{membership}{uncertain}. Candidate scores do not confirm a planet.")
        for variable, value in zip(self.values, [f"{f['period_days']:.4f}",
                f"{f['duration_days']*24:.2f}", f"{f['depth']*1e6:,.0f}", f"{probability:.1%}"]):
            variable.set(value)
        for ax in self.axes:
            ax.clear()
        self.style_axes()
        t, flux, phase = result["time"], result["flux"], result["phase"]
        in_transit = np.abs(phase) < f["duration_days"] / 2
        self.axes[0].scatter(t, flux, s=1, color="#779ac7", alpha=0.55, rasterized=True)
        self.axes[0].scatter(t[in_transit], flux[in_transit], s=3, color=ACCENT)
        limit = min(f["period_days"]/2, max(0.25, f["duration_days"]*2.5))
        visible = np.abs(phase) <= limit
        self.axes[1].scatter(phase[visible]*24, flux[visible], s=2, color="#779ac7", alpha=0.18)
        edges = np.linspace(-limit, limit, 65)
        mids, medians = [], []
        for left, right in zip(edges[:-1], edges[1:]):
            selected = (phase >= left) & (phase < right)
            if selected.any():
                mids.append((left+right)*12)
                medians.append(np.median(flux[selected]))
        self.axes[1].plot(mids, medians, ".-", color=ACCENT, markersize=4, linewidth=1)
        self.axes[1].set_xlim(-limit*24, limit*24)
        self.axes[2].plot(result["period_grid"], result["power"], color=ACCENT, linewidth=0.8)
        self.axes[2].axvline(f["period_days"], color="#ffca76", alpha=0.8)
        self.axes[2].set_ylabel("BLS power", color=MUTED, fontsize=8)
        self.canvas.draw_idle()
        self.status.set("Analysis complete. Export the result or inspect the plots with the toolbar.")

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
        metrics = self.report["metrics"]
        positive = metrics["1"]
        messagebox.showinfo("Real-data evaluation", (
            f"Real TESS sectors {self.report.get('sectors', [self.report['sector']])} / two-minute cadence\n"
            f"Matched observations: {self.report['matched_rows']}\n"
            f"Train / test observations: {self.report['train_rows']} / {self.report['test_rows']}\n"
            f"Train / test distinct stars: {len(self.report['train_tic_ids'])} / {len(self.report['test_tic_ids'])}\n\n"
            f"Accuracy: {metrics['accuracy']:.3f}\nPlanet precision: {positive['precision']:.3f}\n"
            f"Planet recall: {positive['recall']:.3f}\nPlanet F1: {positive['f1-score']:.3f}\n\n"
            f"Decision threshold: {self.report.get('threshold', 0.5):.3f}\n"
            f"Balanced accuracy: {self.report.get('balanced_accuracy', 0):.3f}\n\n"
            f"Confusion matrix [nonplanet, planet]:\n{self.report['confusion_matrix_labels_0_1']}\n\n"
            f"{self.report.get('evaluation_status', '')}\n\n{self.report['limitations']}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="Analyze a held-out star on startup")
    args = parser.parse_args()
    app = Explorer()
    if args.demo:
        app.after(800, app.test_star)
    app.mainloop()
