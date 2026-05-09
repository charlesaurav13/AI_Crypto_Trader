"""Global pytest configuration — environment setup before any imports."""
import os

# Prevent OpenMP deadlock when XGBoost and PyTorch are both loaded in the same
# process. XGBoost initializes the Intel OpenMP runtime (libiomp5); PyTorch uses
# either libgomp or the same libiomp5. Allowing duplicate OpenMP libs avoids the
# inter-library deadlock on macOS.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Single-thread mode for both libraries to sidestep the thread-pool conflict.
os.environ.setdefault("OMP_NUM_THREADS", "1")
