import os
import pandas as pd
import json

def get_single_region_crs(cur_path):
    cur_crs = ""
    if "UTM.json" not in os.listdir(cur_path):
        print("Error: UTM.json not found")
    else:
        with open(os.path.join(cur_path, "UTM.json")) as f:
            data = json.load(f)
            cur_crs = str(data["crs"]).split("::")[-1]
    return [cur_path, cur_crs]

def get_crses(root_path):
    return pd.concat([pd.DataFrame([get_single_region_crs(os.path.join(root_path, region))], columns=['region', 'crs']) for region in os.listdir(root_path)], ignore_index=True)


def get_reg_info(item_path, item,res, project):
    "Получение информации для всех регионов о существующих видах данных, разметке, а также используемой проекции для них"
    region_list = [None] * 6
    for types in os.listdir(item_path):
            ppth = os.path.join(item_path,types)
            if "_разметка" in types:
                region_list[0] = ppth
            if "_Ae" in types:
                dir_ = os.listdir(ppth)[0]
                if os.path.isdir(dir_): 
                    region_list[2] = (os.path.join(ppth,dir_))
                else:
                    region_list[2] = ppth
            if "_Or" in types:
                dir_ = os.listdir(ppth)[0]
                if os.path.isdir(dir_): 
                    region_list[3] = os.path.join(ppth,dir_)
                else:
                    region_list[3] = ppth
            if "карты" in types:
                    region_list[1] = ppth
            if "_SpOr" in types and not "_разметка" in types:
                    region_list[4] = ppth
    flag = False
    for element in region_list:
        if element is not None:
            flag = True
            break
    if flag:
        i=0
        for name in project["region"]:
            if item in name:
                region_list[5] = project.iloc[i,1]
                break
            i+=1
        res.append(region_list.copy())
    if not flag:
        for ch in os.listdir(item_path):
            patth = os.path.join(item_path,ch)
            if os.path.isdir(patth):
                for ress in os.listdir(patth):
                    get_reg_info(patth,item,res, project)


def get_data(data_path):
    res=[]
    for item in os.listdir(data_path):
        item_path = os.path.join(data_path, item)
        get_reg_info(item_path, item,res, get_crses(data_path))
    df = pd.DataFrame(res, columns=["Разметка", "LI","Ae","Or","Spor","UTM"])
    return df


def deal_with_razmetka(instance_data, path):
    dirs = [os.path.join(path, x) for x in os.listdir(path)]
    if dirs is None:
        instance_data[tip]["разметка"] = None
        return
    if any(not os.path.isdir(x) for x in dirs):
        # одна не папка, значит все не папки, у нас один тип и привязываем к нему
        x = list(instance_data.keys())
        x.remove("sub_region_name")
        if len(x) != 1:
            print("Не удалось определить, к чему разметка. Варианты:", x)
        else:
            this_type = x[0]
            instance_data[this_type]["разметка"] = dirs
    else:
        for x in dirs:
            tip = type_speller(list(os.path.split(x))[-1])
            instance_data[tip]["разметка"] = [os.path.join(x, i) for i in os.listdir(x)]
        # папки с разметками, по типу ходим
        pass

def gather_li(path):
    tifs = list(map(lambda x: [int(x.split("_")[0]), x], os.listdir(path)))
    tifs.sort(key=lambda x: x[0])
    batches = []
    cur_batch = []
    is_c = False
    is_ch = False
    is_g = False
    is_i = False
    for img in tifs:
        cur_path = os.path.join(path, img[1])
        if (cur_path.endswith(".tif")):
            tif_type = (img[1].split("_")[-1]).split(".")[0]
        else:
            continue
        if tif_type.lower() == "c":
            if is_c:
                #предыдущий батч неполный, принудительно завершаем
                batches.append(cur_batch)
                cur_batch = []
                is_c = False
                is_ch = False
                is_g = False
                is_i = False
            is_c = True
            cur_batch.append(cur_path)
        if tif_type.lower() == "ch":
            if is_ch:
                #предыдущий батч неполный, принудительно завершаем
                batches.append(cur_batch)
                cur_batch = []
                is_c = False
                is_ch = False
                is_g = False
                is_i = False
            is_c = True
            cur_batch.append(cur_path)
        if tif_type.lower() == "g":
            if is_ch:
                #предыдущий батч неполный, принудительно завершаем
                batches.append(cur_batch)
                cur_batch = []
                is_c = False
                is_ch = False
                is_g = False
                is_i = False
            is_c = True
            cur_batch.append(cur_path)
        if tif_type.lower() == "i":
            if is_ch:
                #предыдущий батч неполный, принудительно завершаем
                batches.append(cur_batch)
                cur_batch = []
                is_c = False
                is_ch = False
                is_g = False
                is_i = False
            is_c = True
            cur_batch.append(cur_path)
        if is_c and is_ch and is_g and is_i:
            # корректное завершение батча
            batches.append(cur_batch)
            cur_batch = []
    if cur_batch:
        #батч предыдущий завершаем
        batches.append(cur_batch)
    return batches

def gather_ae(path):
    batches = []
    for img in os.listdir(path):
        cur_path = os.path.join(path, img)
        if os.path.isfile(cur_path) and cur_path.endswith(".tif"):
            batches.append([cur_path])
        elif os.path.isdir(cur_path):
            return gather_ae(cur_path)
    return batches

def gather_or(path):
    batches = []
    for img in os.listdir(path):
        cur_path = os.path.join(path, img)
        if os.path.isfile(cur_path) and cur_path.endswith(".tif"):
            batches.append([cur_path])
        elif os.path.isdir(cur_path):
            return gather_or(cur_path)
    return batches

def gather_spor(path):
    #path = "dataset\train\086_ВОРОНЕЖ_FINAL\01_Воронеж_1_FINAL\04_Воронеж_1_SpOr"
    #надо [[нужные фотки для воронеж1],[нужные фотки для воронеж2]] или как удобно)
    batches =[]
    for files in os.listdir(path):
        cur_path = os.path.join(path,files)
        if  os.path.isdir(cur_path):
            for sec_file in os.listdir(cur_path):
                sec_path = os.path.join(cur_path,sec_file)
                if  ("Sp_posle_pansharp" in  sec_path and sec_path.endswith(".tif")) or ("Sp_posle_Pansharp" in sec_path and sec_path.endswith(".tif")):
                    batches.append([sec_path])
                if os.path.isdir(sec_path):
                    for third_file in os.listdir(sec_path):
                        third_path = os.path.join(sec_path,third_file)
                        if ("Sp_posle_pansharp" in  third_path and  third_path.endswith(".tif")) or ("Sp_posle_Pansharp" in  third_path and  third_path.endswith(".tif")):
                            batches.append([third_path])

    if len (batches)>0:
        return batches
    for files in os.listdir(path):
        cur_path = os.path.join(path,files)
        if  os.path.isdir(cur_path):
            for sec_file in os.listdir(cur_path):
                sec_path = os.path.join(cur_path,sec_file)
                if  not os.path.isdir(sec_path) and sec_path.endswith(".tif"):
                    batches.append([sec_path])
    if len (batches)>0:
        return batches
    for files in os.listdir(path):
        cur_path = os.path.join(path,files)
        if  not os.path.isdir(cur_path):
             if  not os.path.isdir(cur_path) and cur_path.endswith(".tif"):
                    batches.append([cur_path])

    return batches
    

def get_reg_instance(reg_instance_path):
    instance_data = {}
    instance_data["sub_region_name"] = os.path.split(reg_instance_path)[-1]
    razmetka = None
    for x in os.listdir(reg_instance_path):
        cur_path = os.path.join(reg_instance_path, x)
        if not os.path.isdir(cur_path):
            continue
        tip = type_speller(cur_path)
        if "разметка" in x.lower():
            if razmetka is not None:
                print("Error: вторая разметка, предыдущая", razmetka, ", новая",  x, "будет использована новая")
            razmetka = cur_path
        elif "li" == tip:
            instance_data["li"] = instance_data.get("li", {})
            instance_data["li"]["data"] = instance_data["li"].get("data", []) + gather_li(cur_path)
        elif "ae" == tip:
            instance_data["ae"] = instance_data.get("ae", {})
            instance_data["ae"]["data"] = instance_data["ae"].get("data", []) + gather_ae(cur_path)
        elif "spor" == tip:
            instance_data["spor"] = instance_data.get("spor", {})
            instance_data["spor"]["data"] = instance_data["spor"].get("data", []) + gather_spor(cur_path)
        elif "or" == tip:
            instance_data["or"] = instance_data.get("or", {})
            instance_data["or"]["data"] = instance_data["or"].get("data", []) + gather_or(cur_path)
        
        else:
            if os.path.isdir(cur_path):
                print("Warning: формат папки не определен ", cur_path)
            else:
                print("Warning: формат файла не определен", cur_path)
    if razmetka is None:
        print("Error: разметка не найдена")
    deal_with_razmetka(instance_data, razmetka)
    return instance_data

def type_speller(path):
    # to counter cyrillic symbols
    path = path.lower()
    if "li" in path:
        return "li"
    if "ae" in path or "аe" in path or "aе" in path or "ае" in path:
        return "ae"
    if "spor" in path or "sрor" in path or "spоr" in path or "sроr" in path:
        return "spor"
    if "or" in path or "оr" in path:
        return "or"
    return None

def VORONEZH_CHECK(path):
    path = path.lower()
    if "li" in path:
        return False
    if "ae" in path or "аe" in path or "aе" in path or "ае" in path:
        return False
    if "or" in path or "оr" in path:
        return False
    if "разметка"in path:
        return False
    print("VORONEZH CASE, multiple data dirs", path)
    return True


def get_json(root_path):
    # Проходимся по регионам
    data = []
    for dir in os.listdir(root_path):
        reg_data = {}
        cur_region = os.path.join(root_path, dir)
        if not os.path.isdir(cur_region):
            continue
        region_dirs = list(filter(os.path.isdir, map(lambda x: os.path.join(cur_region, x), os.listdir(cur_region))))
        if "UTM.json" not in os.listdir(cur_region):
            print("ERROR: UTM.json not found")
        else:
            with open(os.path.join(cur_region, "UTM.json")) as f:
                d = json.load(f)
                reg_data["UTM"] = d["crs"].split("::")[-1]
        # дальше у нас проверка, если тут Li\Ae\Or\Spor\разметка, то один инстанс региона, иначе воронеж момент
        is_voronezh = False
        for i in region_dirs:
            if VORONEZH_CHECK(i):
                is_voronezh = True
                break
        if is_voronezh:
            reg_data["region_name"]=dir
            # воронеж1, воронеж2, по путям
            x = []
            for reg_instance in region_dirs:
                x.append(get_reg_instance(reg_instance))
            reg_data["reg_instance"] = x
            
        else:
            reg_data["region_name"]=dir
            x = [get_reg_instance(cur_region)]
            reg_data["reg_instance"] = x
            # просто город и все
        data.append(reg_data)
    return data


if __name__ == "__main__":
    with open('result.json', 'w') as fp:
        json.dump(get_json(".\\dataset\\train"), fp)
    # df = get_data('.\\dataset\\train')
    # df = df.drop_duplicates()
    # df.to_excel('.\\dataset\\file.xlsx', index=False)