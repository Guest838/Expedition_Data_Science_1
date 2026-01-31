import sys
import torch
import torch.nn.functional as F
tile_size = 2048
import shapely
from shapely.geometry import Polygon, mapping
from tools.object_extractor import generate_objects, update_metadata_only, multigeojson_to_multichannel_mask
from tools.data_extractor import get_json
from tools.test_splitter import create_test
from tools.geometry_shit import merge_multiband_windowss
from tools.test_module import mask_to_polygons, apply_transforms
from tools.segformerB0 import SegFormerB0_12Channel
from tools.efficient_net import get_efficientnet_b4_12ch
import json
import os
import rasterio
from tqdm import tqdm
from tools.create_window import create_windows
import logging

threshhold = 0.5

# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)


def infer_pic_type(data, result, models, vis_model = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for class_idx in range(10):
        result = []
        model = models[class_idx]
        if model is None:
            continue
        dataloader = torch.utils.data.DataLoader(data, batch_size=1)
        data_iter = iter(dataloader)
        model.to(device)
        dataloader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True)
        model.eval()
        for obj in tqdm(data_iter):
            # print(f"Количество обработанных изображений {photo_counter}")
            photo_counter+=1
            for batches in range(len(obj["batches"])):
                res = obj["batches"][batches]
                counter += 1
                update_metadata_only(res[0][0], res[0][0], int(obj["UTM"][0]))
                with rasterio.open(res[0][0]) as src:
                    stride = tile_size // 2
                    x_range = range(0, src.width + tile_size, stride)
                    y_range = range(0, src.height + tile_size, stride)
                    # factor_stride_x = random.randint(0, tile_size)
                    # factor_stride_y = random.randint(0, tile_size)
                    #print(x_range, y_range)
                    for y in tqdm(y_range):
                        for x in x_range:
                            window_bounds = (x, y, x + tile_size, y + tile_size)
                            y_end = min(window_bounds[3], int(src.height))
                            x_end = min(window_bounds[2],  int(src.width))
                            image = merge_multiband_windowss(res,x,y,tile_size)
                            image  = F.interpolate(image, scale_factor=0.25, mode='area')
                            image = image.float().to(device)
                            # multimask = multigeojson_to_multichannel_mask(images["vector_mask"],src, window_bounds)
                            if vis_model is not None:
                                eye = torch.softmax(vis_model(image),dim =1)
                                if eye[0][0]>2*eye[0][class_idx] or eye[0][class_idx]<0.2:
                                    continue
                            outputs = model(image)
                            print(outputs.shape)
                            outputs = torch.sigmoid(outputs)
                            outputs = F.interpolate(outputs, scale_factor=4, mode='nearest')
                            for polygon1 in mask_to_polygons(outputs[0][0], x,y, threshhold):#МОЖЕТ СЛОМАТЬСЯ ИЗ_ЗА ФОРМАТА РАСТЕРИО ХРАНЕНИЯ ИЗОБРАЖЕНИЙ (h,w)):
                                polygon = apply_transforms(polygon1, "EPSG:"+ obj["UTM"][0],src.transform)
                                polygons.append(polygon)
                            torch.cuda.empty_cache()
            mask_names = ["kurgany","dorogi","fortifikatsii","arkhitektury","yamy","gorodishche","inoe","selishcha","pashni","mezha"]
            polygons = list(shapely.union_all(polygons).normalize().geoms)
            print(f'Image [{res[0][0]}]')
            feature_reg_name = obj["region_name"][0]
            feature_subreg_name = "" if obj["region_name"][0] == obj["sub_region_name"][0] else obj["sub_region_name"][0]
            feature_subreg_name = feature_subreg_name
            # классы внутри цикла по классам
            feature_markup_type = obj["markup_type"][0]
            feature_original_crs = "urn:ogc:def:crs:EPSG::" + obj["UTM"][0]
            feature_crs = "urn:ogc:def:crs:EPSG::3857"
            # ----------- результат модели записываем
            feature_cnt = 0
            for polygon in polygons:
                feature_class_name = mask_names[class_idx]
                feature = {}
                feature["type"] = "Feature"
                feature["properties"] = {
                    "class_name": feature_class_name,
                    "region_name": feature_reg_name,
                    "sub_region_name": feature_subreg_name,
                    "markup_type": feature_markup_type,
                    "original_crs": feature_original_crs,
                    "crs": feature_crs,
                    "fid": feature_cnt
                }
                feature_cnt += 1
                feature["geometry"] =  mapping(polygon)
                result.append(feature)
    return result


def infer(input_dir, output_dir,  model_li_paths = None, model_ae_paths = None, model_spor_paths =None,  vis_model_path = None):
    li_data, ae_data, spor_data = create_test(input_dir)
    models = [None]*10
    vis_model = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_li_paths is not None:
        for i in range(len(model_li_paths)):
            models[i] =  SegFormerB0_12Channel(num_classes=1)
            checkpoint= torch.load(model_li_paths[i], map_location='cpu')
            models[i].load_state_dict(checkpoint['model_state_dict'])
            # models[i].to(device)

    if vis_model_path is not None:
        vis_model = get_efficientnet_b4_12ch(pretrained=False)
        checkpoint= torch.load(vis_model_path, map_location='cpu')
        vis_model.load_state_dict(checkpoint['model_state_dict'])
        vis_model.to(device)
    li_res = infer_pic_type(li_data, ans, models, vis_model, device)
    if model_ae_paths is not None:
        models = [None]*10
        for i in range(len(model_ae_paths)):
            models[i] =  SegFormerB0_12Channel(num_classes=1)
            checkpoint= torch.load(model_ae_paths[i], map_location='cpu')
            models[i].load_state_dict(checkpoint['model_state_dict'])
            # models[i].to(device)
    ae_res = infer_pic_type(ae_data, ans, models, vis_model, device)
    if model_spor_paths is not None:
        models = [None]*10
        for i in range(len(model_ae_paths)):
            models[i] =  SegFormerB0_12Channel(num_classes=1)
            checkpoint= torch.load(model_spor_paths[i], map_location='cpu')
            models[i].load_state_dict(checkpoint['model_state_dict'])
            # models[i].to(device)
    spor_res = infer_pic_type(spor_data, ans, models, vis_model, device)
    ans = {"type": "FeatureCollection", "features": li_res + ae_res + spor_res}
    output_file = os.path.join(output_dir, "result.geojson")
    try:
        # Шаг 1: Сериализуем весь объект в строку в памяти
        json_content = json.dumps(ans, ensure_ascii=False, indent=2, allow_nan=False)
    except MemoryError:
        logging.warning("Недостаточно памяти, сохраняем без отступов...")
        # Fallback: без отступов занимает меньше памяти
        json_content = json.dumps(ans, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    except Exception as e:
        logging.warning(f"Ошибка сериализации с отступами: {e}, пробуем без отступов...")
        # Fallback: без отступов для надежности
        json_content = json.dumps(ans, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(json_content)  # Атомарная запись всей строки
        logging.info(f"GeoJSON сохранен")
    except Exception as e:
        logging.error(f"Критическая ошибка записи файла GeoJSON: {e}")
        raise



def main():
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    try:
        infer(
            input_dir,
            output_dir,
            model_initialized = True, 
            model_li_paths=['.\\Segformer_b0_2048_1.pth'],
            vis_model_path='.\\Ef_net_det_.pth'
        )
    except Exception as exc:
        sys.exit(1)


if __name__ == "__main__":
    main()
