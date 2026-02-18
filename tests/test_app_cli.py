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

def test_cli_no_flag_shows_placeholder():
    """Verify that running without flags shows the Gradio placeholder."""
    result = subprocess.run(
        [sys.executable, "app.py"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    combined = result.stdout + result.stderr
    assert "Gradio UI" in combined
