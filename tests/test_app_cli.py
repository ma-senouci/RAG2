import subprocess
import sys

def test_cli_sync_flag_exists():
    """Verify that the --sync flag is accepted and produces a sync summary."""
    result = subprocess.run(
        [sys.executable, "app.py", "--sync"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    combined = result.stdout + result.stderr
    assert "SYNC OPERATION SUMMARY" in combined
    assert "Sync complete" in combined

# Note: UI startup tests are removed due to blocking nature of gr.ChatInterface.launch().
# Chat logic is fully covered in test_app_chat.py.
