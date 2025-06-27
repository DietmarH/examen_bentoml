#!/usr/bin/env python3
"""
Development server starter for the admission prediction API.
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def start_server() -> None:
    """Start the BentoML development server."""
    print("🚀 Starting Admission Prediction API Server...")
    print("=" * 50)
    print("Server will be available at: http://localhost:3000")
    print("Available endpoints:")
    print("  - POST /predict_admission")
    print("  - GET  /health_check")
    print("  - GET  /get_model_info")
    print("=" * 50)
    print("Press Ctrl+C to stop the server")
    print()

    try:
        # Start the BentoML server
        cmd = "uv run bentoml serve src.service:AdmissionPredictionService"
        subprocess.run(cmd, shell=True, cwd=PROJECT_ROOT, check=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)  # Exit with error code


if __name__ == "__main__":
    start_server()
    sys.exit(0)
