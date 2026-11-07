# Exact project changes

GitHub publication adds `.gitignore` and `.gitattributes`, initializes Git LFS for
saved models, and documents restoring excluded raw downloads. Runtime caches,
downloaded FITS files, cached sector manifests, and logs are intentionally excluded
from Git; the inventory still records those local scientific data artifacts.

Modified existing source, documentation, dependency, and model/report files:

- app.py
- README.md
- requirements-svm.txt
- requirements-tested.txt
- smoke_test_app.py
- svm_classifier.py
- svm_model.joblib
- SVM_USAGE.md
- tess_pipeline.py
- train_real_tess.py
- training_report.json
- verify_project.py

Added source modules and tests:

- evaluate_model.py
- experiment.py
- model_selection.py
- prefetch_data.py
- prepare_dataset.py
- prepare_followup.py
- report_results.py
- search_pipeline.py
- target_selector.py
- test_pipeline.py
- training_workflow.py
- transit_features.py

Modified existing output artifacts:

- outputs/demo_prediction.csv
- outputs/demo_prediction.png
- outputs/evaluation.png
- outputs/held_out_predictions.csv
- outputs/RESULTS.txt

FILE_MANIFEST.csv lists every added/modified data file, FITS download, feature cache, split/freeze/lock file, model, report, plot, log, and historical backup, with byte size and SHA-256. CHANGES.md and FILE_MANIFEST.csv are new inventory documents and do not hash themselves. Python interpreter caches and unfinished temporary files are excluded.

Original notebooks, exploratory Kepler script, existing raw FITS files, and pilot data tables were preserved. New extraction lives in data/v4 and data/v5; the latter is the final corrected protocol.
