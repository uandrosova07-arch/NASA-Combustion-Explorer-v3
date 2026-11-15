from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data.validate import validate_all
errors=validate_all(ROOT/'data/raw')
if errors:
 print('Validation failed:')
 [print('- '+e) for e in errors]
 raise SystemExit(1)
print('Validation OK: investigations/runs/publications/files/test_matrix links are consistent.')
