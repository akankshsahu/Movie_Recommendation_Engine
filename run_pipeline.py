import subprocess
import sys
import argparse

def run_command(command):
    """Run a shell command in Python and exit on error."""
    print(f"\n>>> Running: {command}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"Error running: {command}")
        sys.exit(result.returncode)

def main(run_app=False):
    # Step 0: Clear Streamlit cache
    run_command("streamlit cache clear")

    # Step 1: Prepare data
    run_command("python -m src.data_prep")

    # Step 2: Train model
    run_command("python -m src.train")

    # Step 3: Evaluate model
    run_command("python -m src.evaluate")

    # Step 4: Optionally launch the Streamlit app
    if run_app:
        print("\n>>> Launching Streamlit app...")
        run_command("streamlit run src/app.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run recommender system pipeline")
    parser.add_argument("--run-app", action="store_true", help="Launch Streamlit app after pipeline")
    args = parser.parse_args()

    main(run_app=args.run_app)
