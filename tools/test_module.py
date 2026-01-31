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
logger = logging.getLogger(__name__)


def round_to_precision(value: float, precision: int = 10) -> float:
    """
    Точное округление float значения до заданного количества знаков после запятой.
    Использует Decimal для избежания погрешностей округления с плавающей точкой.
    
    Args:
        value: Значение для округления
        precision: Количество знаков после запятой (по умолчанию 10)
        
    Returns:
        Округленное значение типа float
    """
    if not isinstance(value, float):
        return value
    
    # Устанавливаем точность для Decimal
    getcontext().prec = 28
    
    # Преобразуем в Decimal, округляем и возвращаем как float
    decimal_obj = Decimal(str(value))
    precision_format = Decimal('0.' + '0' * precision)
    rounded_decimal = decimal_obj.quantize(precision_format)
    return float(rounded_decimal)


# Веса классов (латинские ключи)
CLASS_WEIGHTS: Dict[str, float] = {
    "selishcha": 5.0,
    "kurgany": 4.0,
    "karavannye_puti": 3.0,
    "fortifikatsii": 2.0,
    "gorodishcha": 2.0,
    "arkhitektury": 2.0,
    "pashni": 1.0,
    "dorogi": 1.0,
    "yamy": 1.0,
    "inoe": 0.5,
    "mezha": 0.5,
    # "artefakty_lidara": 0.5 # этот класс не участвует в валидации
}


# Маппинг названий на стандартизированные
CLASS_NAME_MAPPING: Dict[str, str | None] = {
    # Селища
    "селище": "selishcha", "Селище": "selishcha", "селища": "selishcha", "Селища": "selishcha",
    # Пашни
    "пашня": "pashni", "Пашня": "pashni", "пашни": "pashni", "Пашни": "pashni",
    "пахота": "pashni", "pashnya": "pashni", "Pashnya": "pashni",
    "глубин": "pashni", "Глубин": "pashni",
    # Курганы
    "распаханные курганы": "kurgany", "курган": "kurgany", "Курган": "kurgany",
    "курганы": "kurgany", "Курганы": "kurgany", "kurgani": "kurgany", "Kurgani": "kurgany",
    # Караванные пути
    "караванные": "karavannye_puti", "Караванные": "karavannye_puti",
    "караванные пути": "karavannye_puti", "Караванные пути": "karavannye_puti",
    "пути": "karavannye_puti", "Пути": "karavannye_puti",
    # Фортификации
    "фортификация": "fortifikatsii", "Фортификация": "fortifikatsii",
    "фортификации": "fortifikatsii", "Фортификации": "fortifikatsii",
    # Городища
    "городище": "gorodishcha", "Городище": "gorodishcha",
    "городища": "gorodishcha", "Городища": "gorodishcha",
    "gorodishche": "gorodishcha", "Gorodishche": "gorodishcha",
    # Архитектуры
    "архитектура": "arkhitektury", "Архитектура": "arkhitektury",
    "архитектуры": "arkhitektury", "Архитектуры": "arkhitektury",
    # Дороги
    "дорога": "dorogi", "Дорога": "dorogi", "дороги": "dorogi", "Дороги": "dorogi",
    "dorogi": "dorogi", "Dorogi": "dorogi",
    # Ямы
    "яма": "yamy", "Яма": "yamy", "ямы": "yamy", "Ямы": "yamy",
    # Межа
    "межа": "mezha", "Межа": "mezha",
    # Артефакты лидара (игнор)
    "артефакты лидара": None, "Артефакты лидара": None,
    "артефакты_лидара": None, "Артефакты_лидара": None,
    "лидара": None, "артефакт": None, "Артефакт": None,
    # Иное
    "иное": "inoe", "Иное": "inoe", "inoe": "inoe", "Inoe": "inoe",
}


def split_merged_by_source_and_class(geojson_data: Dict[str, Any]) -> Dict[str, Dict[str, Dict]]:
    result: Dict[str, Dict[str, Dict]] = {}
    features = geojson_data.get("features", [])
    for feature in features:
        props = feature.get("properties", {}) or {}
        region_name = props.get("region_name", "default_source")
        sub_region_name = props.get("sub_region_name", "")
        if sub_region_name and sub_region_name.strip():
            source_name = f"{region_name}__{sub_region_name}"
        else:
            source_name = region_name
        class_name = props.get("class_name", "unknown")

        if source_name not in result:
            result[source_name] = {}
        if class_name not in result[source_name]:
            result[source_name][class_name] = {"type": "FeatureCollection", "features": []}

        result[source_name][class_name]["features"].append(feature)
    return result


class F2Calculator:
    def calculate_final_f2(
        self,
        predictions: Dict[str, Dict[str, Dict]],
        ground_truth: Dict[str, Dict[str, Dict]],
        tau: float = 0.5,
        pixel_resolution: float = 1.0,
        min_area_threshold_pixels: float = 3.0,
        annotation_tolerance_pixels: float = 2.0,
        no_fp_penalty_classes: Optional[List[str]] = None,
    ) -> float:
        self._validate_inputs(predictions, ground_truth, tau, pixel_resolution)
        
        # Преобразуем no_fp_penalty_classes в список если None
        if no_fp_penalty_classes is None:
            no_fp_penalty_classes = []

        pred_sources = set(predictions.keys())
        gt_sources = set(ground_truth.keys())

        if len(gt_sources) == 0:
            logger.warning("Ground truth не содержит источников — F2 метрика равна 0")
            return 0.0

        missing_in_predictions = gt_sources - pred_sources
        if missing_in_predictions:
            logger.warning(
                "Предсказания отсутствуют для %d GT-источников: %s",
                len(missing_in_predictions),
                sorted(missing_in_predictions)
            )

        weighted_f2_scores: List[float] = []
        processed_sources: List[str] = []

        for source_name in sorted(gt_sources):
            pred_source = predictions.get(source_name, {})

            if source_name not in predictions:
                logger.debug(
                    "Источник '%s' отсутствует в предсказаниях, используется пустой набор",
                    source_name
                )

            source_f2 = self._calculate_source_weighted_f2(
                pred_source,
                ground_truth[source_name],
                tau,
                pixel_resolution,
                min_area_threshold_pixels,
                annotation_tolerance_pixels,
                no_fp_penalty_classes,
            )
            weighted_f2_scores.append(source_f2)
            processed_sources.append(source_name)

        if weighted_f2_scores:
            final_score = float(np.mean(weighted_f2_scores))
            logger.info(
                "final_score рассчитан по %d источникам: %s",
                len(weighted_f2_scores),
                list(zip(processed_sources, [round(score, 4) for score in weighted_f2_scores]))
            )
            return final_score

        logger.warning("Не удалось рассчитать взвешенные F2 метрики по источникам")
        return 0.0

    def _calculate_source_weighted_f2(
        self,
        pred_source: Dict[str, Dict],
        gt_source: Dict[str, Dict],
        tau: float,
        pixel_resolution: float,
        min_area_threshold_pixels: float,
        annotation_tolerance_pixels: float,
        no_fp_penalty_classes: List[str],
    ) -> float:
        # собрать множество классов (с учетом имен в GeoJSON name при наличии)
        all_classes = set()
        for class_name in pred_source.keys():
            norm = CLASS_NAME_MAPPING.get(class_name, class_name)
            if norm is not None:
                all_classes.add(norm)
        for class_name in gt_source.keys():
            norm = CLASS_NAME_MAPPING.get(class_name, class_name)
            if norm is not None:
                all_classes.add(norm)

        total_weight = 0.0
        total_weighted_f2 = 0.0

        for class_name in all_classes:
            if class_name not in CLASS_WEIGHTS:
                if class_name is None or class_name == "artefakty_lidara":
                    continue
                raise ValueError(
                    f"Неизвестный класс '{class_name}'. Допустимые: {list(CLASS_WEIGHTS.keys())}"
                )
            pred_class = pred_source.get(class_name, {"features": []})
            gt_class = gt_source.get(class_name, {"features": []})

            f2 = self._calculate_class_f2(
                pred_class,
                gt_class,
                tau,
                pixel_resolution,
                min_area_threshold_pixels,
                annotation_tolerance_pixels,
                class_name,
                no_fp_penalty_classes,
            )

            weight = CLASS_WEIGHTS[class_name]
            total_weight += weight
            total_weighted_f2 += weight * f2

        return total_weighted_f2 / total_weight if total_weight > 0 else 0.0

    def _calculate_class_f2(
        self,
        pred_class: Dict,
        gt_class: Dict,
        tau: float,
        pixel_resolution: float,
        min_area_threshold_pixels: float,
        annotation_tolerance_pixels: float,
        class_name: str,
        no_fp_penalty_classes: List[str],
    ) -> float:
        pred_polygons = self._extract_polygons(pred_class)
        gt_polygons = self._extract_polygons(gt_class)

        pred_polygons = [
            p for p in pred_polygons if self._is_above_min_area(p, pixel_resolution, min_area_threshold_pixels)
        ]
        gt_polygons = [
            g for g in gt_polygons if self._is_above_min_area(g, pixel_resolution, min_area_threshold_pixels)
        ]

        if not pred_polygons and not gt_polygons:
            return 0.0

        iou_no_buffer = self._calculate_iou_matrix(pred_polygons, gt_polygons)
        if np.any(iou_no_buffer >= 0.999):
            iou_matrix = iou_no_buffer
        else:
            gt_buffered = [
                self._apply_annotation_tolerance(g, pixel_resolution, annotation_tolerance_pixels)
                for g in gt_polygons
            ]
            iou_matrix = self._calculate_iou_matrix(pred_polygons, gt_buffered)

        tp, fp, fn = self._calculate_tp_fp_fn(iou_matrix, tau)
        
        # Проверяем, нужно ли исключить штраф за FP для этого класса
        fp_for_precision = fp
        if class_name in no_fp_penalty_classes:
            logger.info(f"Класс '{class_name}' исключен из штрафа за FP (было FP={fp}, используется FP=0 для precision)")
            fp_for_precision = 0
        
        precision = tp / (tp + fp_for_precision) if (tp + fp_for_precision) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return self._f2(precision, recall)

    def _f2(self, precision: float, recall: float) -> float:
        if precision + recall == 0:
            return 0.0
        return 5 * (precision * recall) / (4 * precision + recall)

    def _extract_polygons(self, geojson_data: Dict) -> List[Polygon]:
        polygons: List[Polygon] = []
        for feature in geojson_data.get("features", []):
            geometry = feature.get("geometry")
            if not geometry:
                continue
            if geometry.get("type") != "Polygon":
                continue
            coordinates = geometry.get("coordinates", [])
            if not coordinates:
                continue
            try:
                poly = Polygon(coordinates[0], coordinates[1:] if len(coordinates) > 1 else None)
                if not poly.is_valid:
                    poly = make_valid(poly)
                if poly.is_valid and not poly.is_empty:
                    polygons.append(poly)
            except Exception:
                continue
        return polygons

    def _is_above_min_area(self, polygon: Polygon, pixel_resolution: float, min_area_threshold_pixels: float) -> bool:
        min_area_sq_meters = min_area_threshold_pixels * (pixel_resolution ** 2)
        return polygon.area >= min_area_sq_meters

    def _apply_annotation_tolerance(
        self,
        polygon: Polygon,
        pixel_resolution: float,
        annotation_tolerance_pixels: float,
    ) -> Polygon:
        try:
            buffer_size = annotation_tolerance_pixels * pixel_resolution
            buffered = polygon.buffer(buffer_size)
            if not buffered.is_valid:
                buffered = make_valid(buffered)
            return buffered if buffered.is_valid else polygon
        except Exception:
            return polygon

    def _calculate_iou_matrix(self, pred_polygons: List[Polygon], gt_polygons: List[Polygon]) -> np.ndarray:
        if not pred_polygons or not gt_polygons:
            return np.zeros((len(pred_polygons), len(gt_polygons)))
        iou = np.zeros((len(pred_polygons), len(gt_polygons)))
        prepared_gt = [prepared.prep(g) for g in gt_polygons]
        for i, p in enumerate(pred_polygons):
            for j, (g, prep_g) in enumerate(zip(gt_polygons, prepared_gt)):
                if not prep_g.intersects(p):
                    continue
                iou[i, j] = self._polygon_iou(p, g)
        return iou

    def _polygon_iou(self, a: Polygon, b: Polygon) -> float:
        try:
            if not a.is_valid or not b.is_valid:
                return 0.0
            inter = a.intersection(b)
            union = a.union(b)
            if union.area == 0:
                return 0.0
            return inter.area / union.area
        except Exception:
            return 0.0

    def _calculate_tp_fp_fn(self, iou_matrix: np.ndarray, tau: float) -> Tuple[int, int, int]:
        if iou_matrix.size == 0:
            return 0, 0, 0
        num_pred, num_gt = iou_matrix.shape
        used_pred = set()
        used_gt = set()
        tp = 0
        while True:
            masked = iou_matrix.copy()
            for i in used_pred:
                masked[i, :] = 0
            for j in used_gt:
                masked[:, j] = 0
            max_iou = np.max(masked)
            if max_iou < tau:
                break
            i, j = np.unravel_index(np.argmax(masked), masked.shape)
            used_pred.add(i)
            used_gt.add(j)
            tp += 1
        fp = num_pred - tp
        fn = num_gt - tp
        return tp, fp, fn

    def _validate_inputs(self, predictions: Dict, ground_truth: Dict, tau: float, pixel_resolution: float) -> None:
        assert 0 < tau < 1, "Порог IoU должен быть в диапазоне (0, 1)"
        assert pixel_resolution > 0, "Разрешение пикселя должно быть положительным"
        assert isinstance(predictions, dict) and isinstance(ground_truth, dict), "Входные данные должны быть словарями"
        assert len(ground_truth) > 0, "Ground truth не должен быть пустым"
        if len(predictions) == 0:
            logger.warning("Предсказания не содержат источников — итоговый F2 будет равен 0")


def check_engine(preds, multimask):
    preds = preds.cpu()
    if len(np.nonzero(multimask[0])[0])!=0:
        plt.imshow(multimask[0])
        plt.colorbar()
        plt.show()
        plt.imshow(preds[0][0])
        plt.colorbar()
        plt.show()
    elif len(np.nonzero(multimask[1])[0])!=0:
        plt.imshow(multimask[1])
        plt.colorbar()
        plt.show()
        plt.imshow(preds[0][1])
        plt.colorbar()
        plt.show()
    elif len(np.nonzero(multimask[2])[0])!=0:
        plt.imshow(multimask[2])
        plt.colorbar()
        plt.show()
        plt.imshow(preds[0][2])
        plt.colorbar()
        plt.show()
    elif len(np.nonzero(multimask[3])[0])!=0:
        plt.imshow(multimask[3])
        plt.colorbar()
        plt.show()
        plt.imshow(preds[0][3])
        plt.colorbar()
        plt.show()
    elif len(np.nonzero(multimask[4])[0])!=0:
        plt.imshow(multimask[4])
        plt.colorbar()
        plt.show()
        plt.imshow(preds[0][4])
        plt.colorbar()
        plt.show()
    elif len(np.nonzero(multimask[5])[0])!=0:
        plt.imshow(multimask[5])
        plt.colorbar()
        plt.show()
        plt.imshow(preds[0][5])
        plt.colorbar()
        plt.show()
    elif len(np.nonzero(multimask[6])[0])!=0:
        plt.imshow(multimask[6])
        plt.colorbar()
        plt.show()
        plt.imshow(preds[0][6])
        plt.colorbar()
        plt.show()
    elif len(np.nonzero(multimask[7])[0])!=0:
        plt.imshow(multimask[7])
        plt.colorbar()
        plt.show()
        plt.imshow(preds[0][7])
        plt.colorbar()
        plt.show()
    elif len(np.nonzero(multimask[8])[0])!=0:
        plt.imshow(multimask[8])
        plt.colorbar()
        plt.show()
        plt.imshow(preds[0][8])
        plt.colorbar()
        plt.show()
    elif len(np.nonzero(multimask[9])[0])!=0:
        plt.imshow(multimask[9])
        plt.colorbar()
        plt.show()
        plt.imshow(preds[0][9])
        plt.colorbar()
        plt.show()

def apply_transforms(poly: Polygon, source_crs) -> Optional[Polygon]:
    try:
        transformer = Transformer.from_crs(source_crs, "EPSG:3857", always_xy=True)
        poly = shapely_transform(transformer.transform, poly)

        poly = make_valid(poly)
        return poly if not poly.is_empty else None
    except Exception as e:
        print(e)
        return None



def detections_to_features(detections, class_names: Dict[int, str], image_shape: Tuple[int, int],
                           transform: Affine, source_crs: str, scale: float, angle: float,
                           crop_offset: Tuple[int, int], rotation_center: Tuple[float, float],
                           region_name: str, markup_type: str = "li") -> List[Dict[str, Any]]:
    """Converts tiled detections with masks to GeoJSON features."""
    _ = image_shape
    affine_params = (transform.a, transform.b, transform.d, transform.e, transform.c, transform.f)
    transformer = Transformer.from_crs(source_crs, "EPSG:3857", always_xy=True)
    source_urn = f"urn:ogc:def:crs:{source_crs.replace(':', '::')}"
    target_urn = "urn:ogc:def:crs:EPSG::3857"
    features: List[Dict[str, Any]] = []
    rotate_origin = (float(rotation_center[0]), float(rotation_center[1]))
    
    # Pre-compute transformation matrices
    need_rotate = abs(angle) > 1e-3
    need_scale = abs(scale - 1.0) > 1e-6
    
    
    for det in detections:
        mask = getattr(det, "mask", None)
        if mask is None or not mask.bool_mask.any():
            continue
        
        polygons = mask_to_polygons(det.mask.bool_mask.astype(np.uint8))
        for poly in polygons:
            poly = apply_transforms(poly)
            if poly is None or poly.area < 1.0:
                continue
            
            # Aggressive simplification to reduce coordinates
            poly_simplified = poly.simplify(2.0, preserve_topology=True)
            
            features.append({
                "type": "Feature",
                "properties": {
                    "class_name": class_names.get(det.category.id, det.category.name),
                    "region_name": region_name,
                    "sub_region_name": "",
                    "markup_type": markup_type,
                    "original_crs": source_urn,
                    "crs": target_urn,
                    "fid": 0,
                    "confidence": round(det.score.value, 3),
                },
                "geometry": mapping(poly_simplified),
            })
    
    return features

# трансформ из окна в изображение
# трансформ изображения в координаты
# трансформ из коорд в epsg:3587

def mask_to_polygons(mask: np.ndarray, max_polygons: int = 100) -> Iterable[Polygon]:
    """Convert binary mask to polygons using contours (fast) instead of rasterio shapes (slow)."""
    mask_uint8 = (mask>0.5).astype(np.uint8)
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
        coords = [(float(pt[0][0]), float(pt[0][1])) for pt in simplified]
        
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

            
def gaussian_window_2d(shape,device, sigma_factor=1/8, value_scaling_factor=1):
    """
    Создаёт 2D гауссову весовую маску для заданной формы.
    
    Параметры:
        shape (tuple): (H, W) — размер окна.
        sigma_factor (float): доля от размера окна для сигмы (по умолчанию 1/8, как в nnU-Net).
    
    Возвращает:
        np.ndarray: маска весов той же формы, что и shape, нормированная.
    """
    h, w = shape
    impulse = np.zeros((h, w))
    impulse[h // 2, w // 2] = 1.0
    
    sigma = (h * sigma_factor, w * sigma_factor)

    gaussian_importance_map = gaussian_filter(impulse, sigma=sigma, mode='constant', cval=0)
    gaussian_importance_map /= (np.max(gaussian_importance_map) / value_scaling_factor)
    mask = gaussian_importance_map == 0
    gaussian_importance_map[mask] = np.min(gaussian_importance_map[~mask])
    
    return  gaussian_importance_map

    

