"""
QuantTrade ML Pipeline — CLI Main Entry Point
Supports running the entire quantitative trading ML pipeline from start to finish.
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.orchestration.runner import PipelineRunner

def main() -> None:
    parser = argparse.ArgumentParser(description="QuantTrade E2E Quantitative ML Pipeline")
    parser.add_argument(
        "--force",
        action="store_true",
        default=True,  # Default to running the complete pipeline from start to finish
        help="Force re-execution of all pipeline stages"
    )
    parser.add_argument(
        "--no-force",
        dest="force",
        action="store_false",
        help="Use cached outputs if available instead of re-executing steps"
    )
    args = parser.parse_args()

    runner = PipelineRunner(force=args.force)
    runner.run()

if __name__ == "__main__":
    main()
