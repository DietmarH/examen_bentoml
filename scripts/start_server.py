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
    print("🚀 Starting Secure Admission Prediction API Server...")
    print("🔐 Now with JWT Authentication!")
    print("=" * 60)
    print("Server will be available at: http://localhost:3000")
    print("API Documentation: http://localhost:3000/docs")
    print("")
    print("🔑 Demo Users:")
    print("  - admin / admin123 (Administrator)")
    print("  - user / user123 (Regular User)")
    print("  - demo / demo123 (Demo Account)")
    print("")
    print("📡 Available endpoints:")
    print("  - POST /login - Authenticate and get access token")
    print("  - POST /predict - Single prediction (requires auth)")
    print("  - POST /predict_batch - Batch predictions (requires auth)")
    print("  - POST /status - API status (optional auth)")
    print("  - POST /admin_users - List users (admin only)")
    print("  - POST /admin_model_info - Detailed model info (admin only)")
    print("")
    print("🧪 Demo Scripts:")
    print("  - python scripts/demo_secure_api.py (comprehensive demo)")
    print("  - python scripts/demo_api.py (legacy demo)")
    print("=" * 60)
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
