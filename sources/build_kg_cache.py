"""Build the graph caches from the processed CSVs: python sources/build_kg_cache.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from kb_graph import KBGraph

for kb, name in {"ESCO": "occSkill_graph.pkl", "ONET": "onet_occSkill_graph.pkl"}.items():
    KBGraph.from_processed(ROOT / "sources" / kb / "processed").save(ROOT / "cache" / name)
    print("wrote", ROOT / "cache" / name)
