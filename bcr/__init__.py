__version__ = "1.0.0"

import sys
from pathlib import Path

_base_path = Path(__file__).parent.parent
sys.path.append(str(_base_path / "hipt"))
sys.path.append(str(_base_path / "DCTM"))
