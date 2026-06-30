import subprocess
import sys
import os

def run_example_script(script_name):
    # Determine path to the script relative to root
    script_path = os.path.join("examples", script_name)
    assert os.path.exists(script_path), f"Script {script_path} does not exist"
    
    # Run the script using the same Python interpreter with cwd="examples"
    result = subprocess.run(
        [sys.executable, script_name],
        cwd="examples",
        capture_output=True,
        text=True
    )
    
    # Assert return code is 0
    assert result.returncode == 0, f"Script {script_name} failed with exit code {result.returncode}.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

def test_plot_metric():
    run_example_script("plot_metric.py")

def test_plot_redundancy():
    run_example_script("plot_redundancy.py")
