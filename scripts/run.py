"""Source checkout entry for the installed phyloopno command."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from phyloopno.cli import main
if __name__ == '__main__':
    raise SystemExit(main())
