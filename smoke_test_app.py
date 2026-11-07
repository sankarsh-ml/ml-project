"""Run the actual GUI worker/queue/render/export path on a real held-out star."""
import time
import traceback
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
                (ROOT / "outputs").mkdir(exist_ok=True)
                app.save_result(ROOT / "outputs/demo_prediction.csv")
                assert len(app.axes[0].collections) >= 2
                assert len(app.axes[1].lines) > 0
                assert len(app.axes[2].lines) >= 2
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
