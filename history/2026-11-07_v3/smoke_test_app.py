"""Run the actual GUI worker/queue/render/export path on a real held-out star."""
import time
from app import Explorer
from tess_pipeline import ROOT


def main():
    app = Explorer()
    app.withdraw()
    outcome = {}
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
                assert app.result["features"]["tic_id"] in app.report["test_tic_ids"]
                assert "experimental" in app.verdict.get().lower()
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

    app.test_star()
    app.after(100, check)
    app.mainloop()
    assert outcome.get("ok"), outcome.get("error", "GUI closed before verification")
    print("PASS: real FITS -> background analysis -> SVM -> GUI rendering -> CSV/PNG export")


if __name__ == "__main__":
    main()
