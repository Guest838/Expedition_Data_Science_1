import os
import numpy as np
import pandas as pd
from osgeo import gdal, osr
import rasterio
import rasterio.mask
import geopandas as gpd
from shapely.geometry import box
from affine import Affine
from rasterio.windows import Window
from rasterio.features import rasterize
import matplotlib.pyplot as plt
import json
from tools.data_extractor import get_json

def clip_raster_by_geojson(raster_path, geojson_path, output_path=None):
    with rasterio.open(raster_path) as src:
        gdf = gpd.read_file(geojson_path)
        gdf1 =gdf.set_crs(src.crs,allow_override=True)
        gdf = gdf.to_crs(src.crs)
        raster_bounds = src.bounds
        raster_bbox = box(*raster_bounds)
        if len(gdf) > 0:
            geojson_bounds = gdf.total_bounds
            geojson_bbox = box(*geojson_bounds)
            intersection = raster_bbox.intersects(geojson_bbox)
            if not intersection:
                gdf = gdf1
                geojson_bounds = gdf.total_bounds
                geojson_bbox = box(*geojson_bounds)

        valid_geometries = []
        for geom in gdf.geometry:
            if geom is not None and not geom.is_empty:
                if raster_bbox.intersects(geom):
                    valid_geometries.append(geom)
                    
        
        if not valid_geometries:
            return None, None
        try:
            out_image, out_transform = rasterio.mask.mask(
                src, 
                valid_geometries, 
                crop=True, 
                all_touched=True
            )
            
            out_meta = src.meta.copy()
            out_meta.update({
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })

            return out_image
            
        except ValueError as e:
            print(f" Ошибка при маскировании: {e}")
            return None, None

def multigeojson_to_multichannel_mask(
    info,
    raster,
    window_pixel_bounds,
):
    flag = False
    src = raster
    if window_pixel_bounds is not None:
        x1, y1, x2, y2 = window_pixel_bounds
        w, h = x2 - x1, y2 - y1
        left, top = rasterio.transform.xy(src.transform, y1, x1)
        right, bottom = rasterio.transform.xy(src.transform, y2, x2)
        window_bbox = box(left, bottom, right, top)

        window_transform = src.transform * Affine.translation(x1, y1)
        out_shape = (h, w)

    crs = src.crs
    num_classes = 10
    mask = np.zeros((num_classes, out_shape[0], out_shape[1]), dtype=int)
    masks = info
    for path in masks:
        path = path[0]
        try:
            gdf = gpd.read_file(path)
            if gdf.empty:
                continue

            if gdf.crs is None:
                gdf = gdf.set_crs(crs)
            elif gdf.crs != crs:
                gdf1 = gdf.set_crs(src.crs,allow_override=True)
                raster_bounds = src.bounds
                raster_bbox = box(*raster_bounds)
                geojson_bounds = gdf1.total_bounds
                geojson_bbox = box(*geojson_bounds)
                intersection = raster_bbox.intersects(geojson_bbox)
                if intersection:
                    gdf = gdf1
                else:
                    gdf = gdf.to_crs(crs)
            if window_bbox is not None:
                gdf = gdf[gdf.intersects(window_bbox)]
                if gdf.empty:
                    continue
            shapes = []
            for geom in gdf.geometry:
                if geom.is_empty or not geom.is_valid:
                    continue
                shapes.append((geom, 1))
            if shapes:
                binary_mask = rasterize(
                    shapes,
                    out_shape=out_shape,
                    transform=window_transform,
                    fill=0,
                    dtype=int
                )
                fname_lower = os.path.basename(path).lower()
                if "курганы" in fname_lower:
                    mask[0] = binary_mask
                elif "дороги" in fname_lower:
                    mask[1] = binary_mask
                elif "фортификация" in fname_lower or "фортификации" in fname_lower:
                    mask[2] = binary_mask
                elif "архитектура" in fname_lower:
                    mask[3] = binary_mask
                elif "ямы" in fname_lower:
                    mask[4] = binary_mask
                elif "городище" in fname_lower or "городища" in fname_lower:
                    mask[5] = binary_mask
                elif "иное" in fname_lower:
                    mask[6] = binary_mask
                elif "селище" in fname_lower:
                    mask[7] = binary_mask
                elif "пашня" in fname_lower or "пахота" in fname_lower:
                    mask[8] = binary_mask
                elif "межа" in fname_lower:
                    mask[9] = binary_mask
                
        except Exception as e:
            print(f"Ошибка при обработке {path}: {e}")
            continue
    if flag is True:
        plt.imshow(mask[0])
        plt.colorbar()
        plt.show()
        plt.imshow(mask[1])
        plt.colorbar()
        plt.show()
        plt.imshow(mask[2])
        plt.colorbar()
        plt.show()
        plt.imshow(mask[3])
        plt.colorbar()
        plt.show()
        plt.imshow(mask[4])
        plt.colorbar()
        plt.show()
        plt.imshow(mask[5])
        plt.colorbar()
        plt.show()
        plt.imshow(mask[6])
        plt.colorbar()
        plt.show()
        plt.imshow(mask[7])
        plt.colorbar()
        plt.show()
        plt.imshow(mask[8])
        plt.colorbar()
        plt.show()
        plt.imshow(mask[9])
        plt.colorbar()
        plt.show()
    return mask

def get_file_paths_from_directory(root_dir, extensions=None):
    file_paths = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if extensions is None or any(file.endswith(ext) for ext in extensions):
                file_paths.append(os.path.join(root, file))
    return file_paths


def update_metadata_only(input_file, output_file, new_epsg):
    
    dataset = gdal.Open(input_file, gdal.GA_Update)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(new_epsg)
    dataset.SetProjection(srs.ExportToWkt())
    dataset = None


def get_info_objects(dataset):
    '''Создание таблицы про количество объектов классов в каждом из регионов '''
    res = []
    for i in range(dataset.shape[0]):
        str_ = dataset.iloc[i]
        row =[str_["Разметка"]]
        mask = get_file_paths_from_directory(str_["Разметка"],extensions=[".geojson"])
        li = None
        if str_["LI"] is not None:
            li = get_file_paths_from_directory(str_["LI"],extensions=[".tif"])
        ae = None
        if str_["Ae"] is not None:
            ae = get_file_paths_from_directory(str_["Ae"],extensions=[".tif"])
        or_ = None
        if str_["Or"] is not None:
            or_ = get_file_paths_from_directory(str_["Or"],extensions=[".tif"])
        Spor_ = None
        if str_["Spor"] is not None:
            Spor_ = get_file_paths_from_directory(str_["Spor"],extensions=["Sp_posle_Pansharp.tif"])
            if len(Spor_) == 0:
                Spor_ = get_file_paths_from_directory(str_["Spor"],["Sp_posle_pansharp.tif"])
                if len(Spor_) == 0:
                    Spor_ = get_file_paths_from_directory(str_["Spor"],[".tif"])

        if li is not None:
             row =[str_["Разметка"]]
             index=0
             meta = [None]*7
             lidar = [0]*10
             aedata = [0]*10
             ortho = [0]*10
             spore = [0]*10
             for photo in li:
                print(photo)
                if "_g" in photo:
                    meta[0]=photo
                    update_metadata_only(photo,photo, int(str_["UTM"]))
                    clip_img = clip_raster_by_geojson(photo,path)
                    for path in mask:
                        print(path)
                        if "Li" in path or "LI" in path:
                                 if clip_img is not None:
                                     if "курганы" in path.lower():
                                         lidar[0] += len(np.nonzero(clip_img)[0])
                                     elif "дороги" in path.lower():
                                         lidar[1] += len(np.nonzero(clip_img)[0])
                                     elif "фортификация" in path.lower() or "фортификации" in path.lower():
                                         lidar[2] += len(np.nonzero(clip_img)[0])
                                     elif "архитектура" in path.lower():
                                         lidar[3] += len(np.nonzero(clip_img)[0])
                                     elif "ямы" in path.lower():
                                         lidar[4] += len(np.nonzero(clip_img)[0])
                                     elif "городище" in path or "городища" in path:
                                         lidar[5] += len(np.nonzero(clip_img)[0])
                                     elif "иное" in path.lower():
                                         lidar[6] += len(np.nonzero(clip_img)[0])
                                     elif "селище" in path.lower():
                                         lidar[7] += len(np.nonzero(clip_img)[0])
                                     elif "пашня" in path.lower() or "пахота" in path.lower():
                                         lidar[8] += len(np.nonzero(clip_img)[0])
                                     elif "межа" in path.lower():
                                         lidar[9] += len(np.nonzero(clip_img)[0])
                else:
                    if "_ch" in photo or "_сh" in photo:
                        meta[2] = photo
                    if ("_c" in photo or "_с" in photo) and not ("_ch" in photo or "_сh" in photo):
                        meta[1] = photo
                    if "_i" in photo:
                        meta[3] = photo
                    update_metadata_only(photo,photo, int(str_["UTM"]))
                    for path in mask:
                            print(path)
                            if "Li" in path or "LI" in path:
                                clip_img = clip_raster_by_geojson(photo,path)
                                if clip_img is not None:
                                     if "курганы" in path.lower():
                                         lidar[0] = max(len(np.nonzero(clip_img)[0]),lidar[0])
                                     elif "дороги" in path.lower():
                                         lidar[1] = max(len(np.nonzero(clip_img)[0]),lidar[1])
                                     elif "фортификация" in path.lower() or "фортификации" in path.lower():
                                         lidar[2] = max(len(np.nonzero(clip_img)[0]),lidar[2])
                                     elif "архитектура" in path.lower():
                                         lidar[3] = max(len(np.nonzero(clip_img)[0]),lidar[3])
                                     elif "ямы" in path.lower():
                                         lidar[4] = max(len(np.nonzero(clip_img)[0]),lidar[4])
                                     elif "городище" in path.lower() or "городища" in path.lower():
                                         lidar[5] += max(len(np.nonzero(clip_img)[0]),lidar[5])
                                     elif "иное" in path.lower():
                                         lidar[6] += max(len(np.nonzero(clip_img)[0]),lidar[6])
                                     elif "селище" in path.lower():
                                         lidar[7] += max(len(np.nonzero(clip_img)[0]),lidar[7])
                                     elif "пашня" in path.lower() or "пахота" in path.lower():
                                         lidar[8] += max(len(np.nonzero(clip_img)[0]),lidar[8])
                                     elif "межа" in path.lower():
                                         lidar[9] += max(len(np.nonzero(clip_img)[0]),lidar[9])
                index += 1
                if index == 4:
                    row.extend(meta)
                    row.extend(lidar)
                    row.extend(aedata)
                    row.extend(ortho)
                    row.extend(spore)
                    row.append(str_["UTM"])
                    res.append(row)
                    print(row)
                    row =[str_["Разметка"]]
                    index=0
                    meta = [None]*7
                    lidar = [0]*10
                    aedata = [0]*10
                    ortho = [0]*10
                    spore = [0]*10
                    
        if ae is not None:
                index = 0
                meta = [None]*7
                lidar = [0]*10
                aedata = [0]*10
                ortho = [0]*10
                spore = [0]*10
                row =[str_["Разметка"]]
                for photo in ae:
                    meta[4]=photo
                    for path in mask:
                        print(path)
                        if "Ae" in path or "Ае" in path or "Аe" in path or "Aе" in path:
                            print(photo)
                            update_metadata_only(photo,photo, int(str_["UTM"]))
                            clip_img = clip_raster_by_geojson(photo,path)
                            if clip_img is not None:
                                if "курганы" in path.lower():
                                     aedata[0] += len(np.nonzero(clip_img)[0])
                                elif "дороги" in path.lower():
                                     aedata[1] += len(np.nonzero(clip_img)[0])
                                elif "фортификация" in path.lower() or "фортификации" in path.lower():
                                     aedata[2] += len(np.nonzero(clip_img)[0])
                                elif "архитектура" in path.lower():
                                     aedata[3] += len(np.nonzero(clip_img)[0])
                                elif "ямы" in path:
                                     aedata[4] += len(np.nonzero(clip_img)[0])
                                elif "городище" in path.lower() or "городища" in path.lower():
                                     aedata[5] += len(np.nonzero(clip_img)[0])
                                elif "иное" in path.lower():
                                     aedata[6] += len(np.nonzero(clip_img)[0])
                                elif "селище" in path.lower():
                                     aedata[7] += len(np.nonzero(clip_img)[0])
                                elif "пашня" in path.lower() or "пахота" in path.lower():
                                     aedata[8] += len(np.nonzero(clip_img)[0])
                                elif "межа" in path.lower():
                                     aedata[9] += len(np.nonzero(clip_img)[0])
                index +=1
                if index == 1:
                    row.extend(meta)
                    row.extend(lidar)
                    row.extend(aedata)
                    row.extend(ortho)
                    row.extend(spore)
                    row.append(str_["UTM"])
                    res.append(row)
                    print(row)
                    row =[str_["Разметка"]]
                    index=0
                    meta = [None]*7
                    lidar = [0]*10
                    aedata = [0]*10
                    ortho = [0]*10
                    spore = [0]*10
        if Spor_ is not None:
                row =[str_["Разметка"]]
                meta = [None]*7
                lidar = [0]*10
                aedata = [0]*10
                ortho = [0]*10
                spore = [0]*10
                for photo in Spor_:
                    meta[5]=photo
                    index=0
                    update_metadata_only(photo,photo, int(str_["UTM"]))
                    for path in mask:
                        print(path)
                        if "SpOr" in path or ("Or" not in path and "Ae" not in path and "Ae" not in path and "Li"  not in path and "LI" not in path):
                             clip_img = clip_raster_by_geojson(photo,path)
                             if clip_img is not None:
                                if "курганы" in path.lower():
                                      spore[0] += len(np.nonzero(clip_img)[0])
                                elif "дороги" in path.lower():
                                      spore[1] += len(np.nonzero(clip_img)[0])
                                elif "фортификация" in path.lower() or "фортификации" in path.lower():
                                     spore[2] += len(np.nonzero(clip_img)[0])
                                elif "архитектура" in path.lower():
                                     spore[3] += len(np.nonzero(clip_img)[0])
                                elif "ямы" in path:
                                     spore[4] += len(np.nonzero(clip_img)[0])
                                elif "городище" in path.lower() or "городища" in path.lower():
                                     spore[5] += len(np.nonzero(clip_img)[0])
                                elif "иное" in path.lower():
                                    spore[6] += len(np.nonzero(clip_img)[0])
                                elif "селище" in path.lower():
                                    spore[7] += len(np.nonzero(clip_img)[0])
                                elif "пашня" in path.lower() or "пахота" in path.lower():
                                    spore[8] += len(np.nonzero(clip_img)[0])
                                elif "межа" in path.lower():
                                    spore[9] += len(np.nonzero(clip_img)[0])
                    index +=1
                    if index == 1:
                        row.extend(meta)
                        row.extend(lidar)
                        row.extend(aedata)
                        row.extend(ortho)
                        row.extend(spore)
                        row.append(str_["UTM"])
                        res.append(row)
                        print(row)
                        row =[str_["Разметка"]]
                        index=0
                        meta = [None]*7
                        lidar = [0]*10
                        aedata = [0]*10
                        ortho = [0]*10
                        spore = [0]*10
                        
        if or_ is not None:
                row =[str_["Разметка"]]
                meta = [None]*7
                lidar = [0]*10
                aedata = [0]*10
                ortho = [0]*10
                spore = [0]*10
                for photo in or_:
                    meta[6]=photo
                    index=0
                    update_metadata_only(photo,photo, int(str_["UTM"]))
                    for path in mask:
                        if "Or" in path and "SpOr" not in path:
                            clip_img = clip_raster_by_geojson(photo,path)
                            if clip_img is not None:
                                if "курганы" in path.lower():
                                     ortho[0] += len(np.nonzero(clip_img)[0])
                                elif "дороги" in path.lower():
                                     ortho[1] += len(np.nonzero(clip_img)[0])
                                elif "фортификация" in path.lower() or "фортификации" in path.lower():
                                     ortho[2] += len(np.nonzero(clip_img)[0])
                                elif "архитектура" in path.lower():
                                     ortho[3] += len(np.nonzero(clip_img)[0])
                                elif "ямы" in path.lower():
                                     ortho[4] += len(np.nonzero(clip_img)[0])
                                elif "городище" in path.lower() or "городища" in path.lower():
                                     ortho[5] += len(np.nonzero(clip_img)[0])
                                elif "иное" in path.lower():
                                     ortho[6] += len(np.nonzero(clip_img)[0])
                                elif "селище" in path.lower():
                                     ortho[7] += len(np.nonzero(clip_img)[0])
                                elif "пашня" in path.lower() or "пахота" in path.lower():
                                     ortho[8] += len(np.nonzero(clip_img)[0])
                                elif "межа" in path.lower():
                                     ortho[9] += len(np.nonzero(clip_img)[0])
                    index +=1
                    if index == 1:
                        row.extend(meta)
                        row.extend(lidar)
                        row.extend(aedata)
                        row.extend(ortho)
                        row.extend(spore)
                        row.append(str_["UTM"])
                        res.append(row)
                        print(row)
                        row =[str_["Разметка"]]
                        index=0
                        meta = [None]*7
                        lidar = [0]*10
                        aedata = [0]*10
                        ortho = [0]*10
                        spore = [0]*10
                        
    df = pd.DataFrame(res, columns=["Разметка", "Photo_g","Photo_c","Photo_ch","Photo_i","Ae","Spor","Or","LI_курганы","LI_дороги","LI_фортификации","LI_архитектура","LI_ямы","LI_городища","LI_иное",
                                    "LI_селище", "LI_пашня","LI_межа", "Ae_курганы","Ae_дороги","Ae_фортификации","Ae_архитектура","Ae_ямы",
                                    "Ae_городища","Ae_иное", "Ae_селище", "Ae_пашня","Ae_межа","Or_курганы","Or_дороги","Or_фортификации",
                                    "Or_архитектура","Or_ямы", "Or_городища","Or_иное", "Or_селище", "Or_пашня","Or_межа","SpOr_курганы",
                                    "SpOr_дороги","SpOr_фортификации","SpOr_архитектура","SpOr_ямы", "SpOr_городища","SpOr_иное", 
                                    "SpOr_селище", "SpOr_пашня","SpOr_межа","UTM"])
    return df



def generate_objects(root_path):
    gen_json = get_json(root_path)
    with open('result.json', 'w') as fp:
        json.dump(gen_json, fp)
    li_data = []
    ae_data = []
    sp_or_data = []
    for regions in gen_json:
        reg_name = regions["region_name"]
        UTM = regions["UTM"]
        for sub_reg in regions["reg_instance"]:
            subreg_name = sub_reg["sub_region_name"]
            if "li" in sub_reg.keys():
                print(sub_reg)
                row = {"markup_type": "li", "UTM": UTM, "region_name": reg_name, "sub_region_name": subreg_name, "batches": sub_reg["li"]["data"], "vector_mask": sub_reg["li"]["разметка"]} # без мержинга 3*4 в 12
                # row = {"UTM": UTM, "region_name": reg_name, "sub_region_name": subreg_name, "batches": sum(sub_reg["li"]["data"], start=[]), "vector_mask": sub_reg["li"]["разметка"]} # с мержингом
                li_data.append(row)
            if "ae" in sub_reg.keys():
                print(sub_reg["ae"]["data"])
                row = {"markup_type": "ae", "UTM": UTM, "region_name": reg_name, "sub_region_name": subreg_name, "batches": sum(sub_reg["ae"]["data"], start=[]), "vector_mask": sub_reg["ae"]["разметка"]}
                ae_data.append(row)
            if "spor" in sub_reg.keys():
                row = {"markup_type": "spor", "UTM": UTM, "region_name": reg_name, "sub_region_name": subreg_name, "batches": sum(sub_reg["spor"]["data"], start=[]), "vector_mask": sub_reg["spor"]["разметка"]}
                sp_or_data.append(row)
            if "or" in sub_reg.keys():
                row = {"markup_type": "or", "UTM": UTM, "region_name": reg_name, "sub_region_name": subreg_name, "batches": sum(sub_reg["or"]["data"], start=[]), "vector_mask": sub_reg["or"]["разметка"]}
                sp_or_data.append(row)
    return li_data, ae_data, sp_or_data


    return gen_json 

if __name__ == "__main__":
    a,b,c = generate_objects(".\\dataset\\train")
    print(a)
    print(b)
    print(c)
