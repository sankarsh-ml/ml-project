"""Run the actual GUI worker/queue/render/export path on a real held-out star."""
import time
import traceback
import numpy as np
from app import Explorer
from tess_pipeline import ROOT
from target_selector import EXAMPLES


def main():
    app = Explorer()
    app.withdraw()
    outcome = {}
    callback_errors = []
    app.report_callback_exception = lambda kind, error, trace: callback_errors.append(
        "".join(traceback.format_exception(kind, error, trace)))
    started = time.monotonic()

    def check():
        if app.busy:
            if time.monotonic() - started > 120:
                outcome["error"] = "GUI analysis timed out"
                app.destroy()
            else:
                app.after(100, check)
        else:
            try:
                assert app.result is not None, app.detail.get()
                assert app.result["features"]["tic_id"] in EXAMPLES
                assert app.verdict.get() in ["Planet-like candidate", "False-positive-like signal"]
                assert "accuracy" not in app.detail.get().lower()
                assert "35%" not in app.detail.get()
                assert app.result["uncertainty"]["resamples"] == 20
                assert "Bootstrap" not in app.detail.get()
                assert "20 resampled fits" in app.measurement_details()
                assert "period_days_bootstrap_std" in app.result["prediction"]
                assert len(app.values) == 3
                assert app.values[0].get().endswith(" days")
                assert app.values[1].get().endswith(" hours")
                assert app.values[2].get().endswith("%")
                shown_depth = float(app.values[2].get().rstrip("%"))
                assert abs(shown_depth-app.result["features"]["depth"]*100) <= .000051
                labels = []
                def inspect_labels(widget):
                    for child in widget.winfo_children():
                        if child.winfo_class() == "Label":
                            labels.append(child.cget("text"))
                        inspect_labels(child)
                inspect_labels(app)
                assert not any("probability" in text.lower() or "not validated" in text.lower() for text in labels)
                (ROOT / "outputs").mkdir(exist_ok=True)
                app.save_result(ROOT / "outputs/demo_prediction.csv")
                assert len(app.axes[0].collections) >= 2
                assert len(app.axes[1].lines) > 0
                assert len(app.axes[2].lines) >= 2
                brightness = np.asarray(app.axes[0].collections[0].get_offsets()[:, 1])
                assert abs(np.median(brightness)-100) < .1
                aligned = np.asarray(app.axes[1].collections[0].get_offsets()[:, 1])
                assert abs(np.median(aligned)) < .1
                assert not app.axes[1].yaxis.get_major_formatter().get_useOffset()
                # A presentation-only fixture checks that a known planet is not
                # silently presented as a catalog false positive when ML errs.
                original = app.result
                preview = dict(original)
                preview["features"] = {**original["features"], "tic_id": "261136679"}
                preview["prediction"] = original["prediction"].copy()
                preview["prediction"]["predicted_label"] = 0
                preview["prediction"]["planet_candidate_probability"] = .49
                app.render(preview)
                assert "disagrees with the catalog" in app.detail.get()
                assert "Borderline prediction" in app.detail.get()
                preview["prediction"]["predicted_label"] = -1
                preview["prediction"]["planet_candidate_probability"] = np.nan
                app.render(preview)
                assert app.verdict.get() == "Insufficient periodic signal"
                outcome["ok"] = True
            except Exception as error:
                outcome["error"] = str(error)
            finally:
                app.destroy()

    # The editable selector accepts arbitrary new IDs and catalog prefixes.
    app.tic.set("TIC 9999999999999999")
    app.tic_selector.accept()
    assert app.tic.get() == "9999999999999999"
    assert "New TIC" in app.target_hint.get()
    app.tic.set("TIC 231670397")
    app.tic_selector.accept()
    assert app.tic.get() == "231670397"
    assert "1" in app.sector_selector.cget("values")
    app.test_star()
    app.after(100, check)
    app.mainloop()
    assert outcome.get("ok"), outcome.get("error", "GUI closed before verification")
    assert not callback_errors, "\n".join(callback_errors)
    print("PASS: real FITS -> background analysis -> selected classifier -> GUI rendering -> CSV/PNG export")


if __name__ == "__main__":
    main()
