"""
Skeleton for aggregating tile predictions into a slide-level heatmap.
Expected CSV columns: x, y, prob_tumor
"""
import numpy as np
import pandas as pd

def tiles_to_heatmap(df: pd.DataFrame, tile_size: int, slide_w: int, slide_h: int):
    heat = np.zeros((slide_h, slide_w), dtype=np.float32)
    for _, r in df.iterrows():
        x, y, p = int(r["x"]), int(r["y"]), float(r["prob_tumor"])
        heat[y:y+tile_size, x:x+tile_size] = p
    return heat
