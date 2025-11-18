import subprocess
import time
import pytest
import requests


@pytest.fixture(scope="session", autouse=True)
def start_api():
    """
    Start the FastAPI server before running API tests.
    Runs on http://localhost:8000
    """
    # Launch API
    process = subprocess.Popen(
        ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait until API is alive
    for _ in range(30):
        try:
            requests.get("http://localhost:8000/health")
            print("API READY!")
            break
        except Exception:
            time.sleep(1)

    yield

    # Stop server at the end
    process.terminate()
