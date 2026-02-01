from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import numpy as np
from decimal import Decimal, getcontext
import sys
import time
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple
import cv2
import rasterio
import torch
from PIL import Image
from affine import Affine
from pyproj import Transformer
from scipy.ndimage import gaussian_filter
from rasterio import features as rio_features
from shapely.affinity import affine_transform as shapely_affine_transform
from shapely.affinity import rotate as shapely_rotate
from shapely.affinity import scale as shapely_scale
from shapely.affinity import translate as shapely_translate
from shapely.geometry import MultiPolygon, mapping, shape
from shapely.ops import transform as shapely_transform
from shapely.ops import unary_union
from types import SimpleNamespace
try:
    from shapely.geometry import Polygon
    from shapely.validation import make_valid
    from shapely import prepared
except ImportError:
    raise ImportError("Требуется библиотека shapely. Установите её: pip install shapely")


#updated funtions from baseline

def apply_transforms(poly: Polygon, source_crs, transform: Affine) -> Optional[Polygon]:
    try:
        transformer = Transformer.from_crs(source_crs, "EPSG:3857")
        affine_params = (transform.a, transform.b, transform.d, transform.e, transform.c, transform.f)
        poly = shapely_affine_transform(poly, affine_params)
        poly = shapely_transform(transformer.transform, poly)

        poly = make_valid(poly)
        return poly if not poly.is_empty else None
    except Exception as e:
        print(e)
        return None


def mask_to_polygons(mask: np.ndarray, y,x,threshold,max_polygons: int = 100) -> Iterable[Polygon]:
    mask_uint8 = (mask>threshold).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    count = 0
    for contour in contours:
        if count >= max_polygons:
            break
        
        # Simplify contour to reduce vertices
        epsilon = 0.5  # Adjust for tolerance
        simplified = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(simplified) < 3:
            continue
        
        # Convert contour to polygon coordinates
        coords = [(float(pt[0][0]+y), float(pt[0][1]+x)) for pt in simplified]
        
        try:
            poly = Polygon(coords)
            poly = make_valid(poly)
            
            if poly.is_empty or poly.area < 0.1:
                continue
            
            if isinstance(poly, MultiPolygon):
                for sub in poly.geoms:
                    if not sub.is_empty:
                        yield sub
                        count += 1
            else:
                yield poly
                count += 1
        except Exception:
            continue

#extra functionality
def gaussian_window_2d(shape,device, sigma_factor=1/8, value_scaling_factor=1):
    h, w = shape
    impulse = np.zeros((h, w))
    impulse[h // 2, w // 2] = 1.0
    
    sigma = (h * sigma_factor, w * sigma_factor)

    gaussian_importance_map = gaussian_filter(impulse, sigma=sigma, mode='constant', cval=0)
    gaussian_importance_map /= (np.max(gaussian_importance_map) / value_scaling_factor)
    mask = gaussian_importance_map == 0
    gaussian_importance_map[mask] = np.min(gaussian_importance_map[~mask])
    
    return  gaussian_importance_map

    

