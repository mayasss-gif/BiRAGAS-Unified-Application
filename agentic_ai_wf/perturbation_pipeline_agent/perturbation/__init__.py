# import os

# # Force a non-GUI backend for batch runs to avoid Tkinter thread errors.
# if not os.environ.get("MPLBACKEND"):
#     os.environ["MPLBACKEND"] = "Agg"
# try:
#     import matplotlib
#     matplotlib.use("Agg", force=True)
# except Exception:
#     pass

# from .run_full_pipeline import run_full_pipeline
from .run_full_pipeline_optimized import run_full_pipeline