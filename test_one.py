import os
import shutil
import yaml
import glob
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def load_class_names(yaml_path):
    with open(yaml_path, 'r',encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data['names']

def parse_label_file(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()
    labels = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        class_id = int(parts[0])
        box = list(map(float, parts[1:]))
        labels.append((class_id, box))
    return labels

def save_label_file(label_path, labels):
    with open(label_path, 'w') as f:
        for class_id, box in labels:
            box_str = ' '.join(f'{b:.6f}' for b in box)
            f.write(f'{class_id} {box_str}\n')

def process_image_label_pair(img_path, label_path, out_img_path, out_lbl_path, desired_class_ids):
    if not label_path.exists():
        return

    labels = parse_label_file(label_path)
    filtered = [(cls_id, box) for cls_id, box in labels if cls_id in desired_class_ids]

    if not filtered:
        return

    save_label_file(out_lbl_path, filtered)
    shutil.copy(img_path, out_img_path)

def filter_dataset(images_dir, labels_dir, yaml_path, desired_class_names, output_dir, max_workers=8):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)

    class_names = load_class_names(yaml_path)
    class_name_to_id = {name: idx for idx, name in enumerate(class_names)}
    desired_class_ids = [class_name_to_id[name] for name in desired_class_names]

    is_subdir = any((images_dir / sub).is_dir() for sub in ['train', 'val', 'test'])
    subsets = ['train', 'val', 'test'] if is_subdir else ['']

    for subset in subsets:
        img_dir = images_dir / subset if subset else images_dir
        lbl_dir = labels_dir / subset if subset else labels_dir
        out_img_dir = output_dir / 'images' / subset if subset else output_dir / 'images'
        out_lbl_dir = output_dir / 'labels' / subset if subset else output_dir / 'labels'
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        img_files = list(img_dir.glob('*.jpg'))
        tasks = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for img_path in tqdm(img_files, desc=f'Processing {subset or "all"} set'):
                label_path = lbl_dir / f'{img_path.stem}.txt'
                out_img_path = out_img_dir / img_path.name
                out_lbl_path = out_lbl_dir / label_path.name

                task = executor.submit(
                    process_image_label_pair,
                    img_path, label_path,
                    out_img_path, out_lbl_path,
                    desired_class_ids
                )
                tasks.append(task)

    print("✅ 数据筛选完成")

##########################################################################
def remap_all_labels(labels_root, original_yaml_path, new_class_names):
    """
    批量遍历 labels 目录，重映射所有标签文件的类别 ID。
    支持 labels/train/, labels/val/, labels/test/ 结构，也支持扁平结构。
    """
    labels_root = Path(labels_root)
    subsets = ['train', 'val', 'test']
    has_subdirs = any((labels_root / sub).is_dir() for sub in subsets)

    # 读取原始 yaml 类别映射
    with open(original_yaml_path, 'r', encoding='utf-8') as f:
        original_yaml = yaml.safe_load(f)
    original_names = original_yaml['names']
    orig_id_to_name = {i: name for i, name in enumerate(original_names)}
    new_name_to_id = {name: i for i, name in enumerate(new_class_names)}

    def process_label_file(label_file):
        with open(label_file, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            old_id = int(parts[0])
            bbox = parts[1:]

            class_name = orig_id_to_name.get(old_id)
            if class_name not in new_name_to_id:
                continue

            new_id = new_name_to_id[class_name]
            new_lines.append(f"{new_id} {' '.join(bbox)}")

        # 覆盖写回
        with open(label_file, 'w') as f:
            for line in new_lines:
                f.write(line + '\n')

    # 遍历标签文件
    if has_subdirs:
        for sub in subsets:
            label_dir = labels_root / sub
            if not label_dir.exists():
                continue
            for label_file in tqdm(label_dir.glob("*.txt"), desc=f"Remapping {sub}"):
                process_label_file(label_file)
    else:
        for label_file in tqdm(labels_root.glob("*.txt"), desc="Remapping flat labels"):
            process_label_file(label_file)

    print("✅ 所有标签文件重映射完成")


def merge_single_file(txt_file, folder_a, folder_b, output_folder, classes_a, classes_b, final_classes):
    """
    合并单个YOLO标签文件。
    
    Args:
        txt_file (str): 输入的txt文件路径
        folder_a (str): 第一个标签文件夹路径
        folder_b (str): 第二个标签文件夹路径
        output_folder (str): 输出合并后的标签文件夹路径
        classes_a (list): 文件夹A的类名列表
        classes_b (list): 文件夹B的类名列表
        final_classes (list): 最终输出的类名顺序
    """
    # 创建类名到最终ID的映射
    class_to_final_id = {cls: idx for idx, cls in enumerate(final_classes)}
    
    # 获取文件名
    filename = os.path.basename(txt_file)
    output_path = os.path.join(output_folder, filename)
    
    # 存储合并后的标注
    merged_annotations = []
    
    # 处理文件夹A的标注
    txt_a_path = os.path.join(folder_a, filename)
    if os.path.exists(txt_a_path):
        with open(txt_a_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    class_name = classes_a[class_id]
                    new_class_id = class_to_final_id[class_name]
                    # 替换类ID，保留其他数据
                    merged_annotations.append(f"{new_class_id} {' '.join(parts[1:])}")
    
    # 处理文件夹B的标注
    txt_b_path = os.path.join(folder_b, filename)
    if os.path.exists(txt_b_path):
        with open(txt_b_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    class_name = classes_b[class_id]
                    new_class_id = class_to_final_id[class_name]
                    # 替换类ID，保留其他数据
                    merged_annotations.append(f"{new_class_id} {' '.join(parts[1:])}")
    
    # 写入合并后的标注
    with open(output_path, 'w') as f:
        for annotation in merged_annotations:
            f.write(annotation + '\n')

def merge_yolo_labels(folder_a, folder_b, output_folder, classes_a, classes_b, final_classes, max_workers=4):
    """
    合并两个YOLO标签文件夹的标注数据，使用多线程和进度条。
    
    Args:
        folder_a (str): 第一个标签文件夹路径
        folder_b (str): 第二个标签文件夹路径
        output_folder (str): 输出合并后的标签文件夹路径
        classes_a (list): 文件夹A的类名列表
        classes_b (list): 文件夹B的类名列表
        final_classes (list): 最终输出的类名顺序
        max_workers (int): 线程池的最大线程数
    """
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取所有唯一的txt文件名
    txt_files_a = set(glob.glob(os.path.join(folder_a, "*.txt")))
    txt_files_b = set(glob.glob(os.path.join(folder_b, "*.txt")))
    all_txt_files = list(txt_files_a.union(txt_files_b))
    
    # 使用多线程处理文件合并
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 使用tqdm显示进度条
        list(tqdm(
            executor.map(
                lambda txt_file: merge_single_file(
                    txt_file, folder_a, folder_b, output_folder, classes_a, classes_b, final_classes
                ),
                all_txt_files
            ),
            total=len(all_txt_files),
            desc="Merging YOLO labels"
        ))
#########################################################################

def process_label_file(args):
    """处理单个标注文件，将A的标注写入B"""
    a_label_path, b_label_path, a_to_b_class_map = args
    if not a_label_path.exists():
        return f"跳过 {a_label_path}，文件不存在"
    
    # 读取A的标注
    with open(a_label_path, 'r') as f:
        a_lines = f.readlines()
    
    # 转换A的标注
    converted_lines = []
    for line in a_lines:
        parts = line.strip().split()
        if not parts:
            continue
        try:
            class_id = int(parts[0])
            if class_id in a_to_b_class_map:
                # 替换类别ID，保留其他字段
                new_class_id = a_to_b_class_map[class_id]
                converted_line = f"{new_class_id} {' '.join(parts[1:])}\n"
                converted_lines.append(converted_line)
        except ValueError:
            return f"错误：{a_label_path} 包含无效的类别ID"
    
    # 如果B的标注文件不存在，直接写入
    if not b_label_path.exists():
        with open(b_label_path, 'w') as f:
            f.writelines(converted_lines)
        return f"创建并写入 {b_label_path}"
    
    # 读取B的现有标注
    with open(b_label_path, 'r') as f:
        b_lines = f.readlines()
    
    # 合并标注（追加A的标注）
    b_lines.extend(converted_lines)
    
    # 写入B的标注文件
    with open(b_label_path, 'w') as f:
        f.writelines(b_lines)
    
    return f"更新 {b_label_path}，追加 {len(converted_lines)} 条标注"

def merge_yolo_labels(dataset_a_path, dataset_b_path, a_classes, b_classes, max_workers=4):
    """将A标注集的标注写入B标注集"""
    # 类名到ID的映射
    a_class_to_id = {name: idx for idx, name in enumerate(a_classes)}
    b_class_to_id = {name: idx for idx, name in enumerate(b_classes)}
    
    # A到B的类别ID映射（只映射A和B的共有类）
    a_to_b_class_map = {
        a_class_to_id[cls]: b_class_to_id[cls]
        for cls in a_classes if cls in b_class_to_id
    }
    
    if not a_to_b_class_map:
        print("错误：A和B的类名没有交集，无法合并")
        return
    
    # 定义目录
    a_label_dir = Path(dataset_a_path) / "labels"
    b_label_dir = Path(dataset_b_path) / "labels"
    subdirs = ["train", "test", "val"]
    
    for subdir in subdirs:
        a_label_subdir = a_label_dir / subdir
        b_label_subdir = b_label_dir / subdir
        
        if not a_label_subdir.exists():
            print(f"A的子目录 {subdir} 不存在，跳过")
            continue
        if not b_label_subdir.exists():
            print(f"B的子目录 {subdir} 不存在，创建目录")
            b_label_subdir.mkdir(parents=True, exist_ok=True)
        
        # 获取A的标注文件
        a_label_paths = list(a_label_subdir.glob("*.txt"))
        if not a_label_paths:
            print(f"{subdir} 中没有txt文件，跳过")
            continue
        
        print(f"处理 {subdir} 子目录（{len(a_label_paths)} 个标注文件）")
        
        # 准备任务列表
        tasks = [(a_label_path, b_label_subdir / a_label_path.name, a_to_b_class_map)
                 for a_label_path in a_label_paths]
        
        # 使用多线程处理
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(
                executor.map(process_label_file, tasks),
                total=len(tasks),
                desc=f"处理 {subdir}"
            ))
        
        # 打印处理结果
        for result in results:
            print(result)

############################################################################
def process_image(img_path, label_subdir):
    """处理单个图片文件，检查是否有对应的txt文件"""
    txt_name = img_path.stem + ".txt"
    txt_path = label_subdir / txt_name
    
    if not txt_path.exists():
        img_path.unlink()
        return f"删除 {img_path}，因为没有对应的 {txt_name}"
    return f"保留 {img_path}，找到对应的 {txt_name}"

def filter_yolo_dataset(dataset_path, max_workers=4):
    """过滤YOLO数据集，删除没有对应txt的jpg文件"""
    image_dir = Path(dataset_path) / "images"
    label_dir = Path(dataset_path) / "labels"
    subdirs = ["train", "test", "val"]
    
    for subdir in subdirs:
        img_subdir = image_dir / subdir
        label_subdir = label_dir / subdir
        
        if not img_subdir.exists() or not label_subdir.exists():
            print(f"子目录 {subdir} 不存在，跳过")
            continue
        
        # 获取所有jpg文件
        img_paths = list(img_subdir.glob("*.jpg"))
        if not img_paths:
            print(f"{subdir} 中没有jpg文件，跳过")
            continue
        
        print(f"处理 {subdir} 子目录（{len(img_paths)} 张图片）")
        
        # 使用多线程处理图片
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 使用tqdm显示进度条
            results = list(tqdm(
                executor.map(lambda p: process_image(p, label_subdir), img_paths),
                total=len(img_paths),
                desc=f"处理 {subdir}"
            ))
        
        # 打印处理结果
        for result in results:
            print(result)



###############################################################################



##############################################################################
# Example usage
if __name__ == '__main__':

    # # ✅ 修改这些参数为你自己的路径和类别
    # images_dir = 'E:/my_yolo/my_yolo//val2017/val2017'  # 原始图片路径（可以是images/train 或 images/）
    # labels_dir = 'E:/my_yolo/my_yolo/coco_dog_cat/labels'  # 标签路径
    # yaml_path = 'E:/machine_study/yolov5-6.0/data/coco.yaml'  # YAML 类别文件
    # desired_class_names = ['dog', 'cat']  # 想要保留的类别
    # output_dir = 'E:/my_yolo/my_yolo/coco_dog_cat'  # 输出目录
    # max_workers = 2  # 并行线程数

    # 复制出指定类的图像和标签目录结构
    # filter_dataset(
    #     images_dir=images_dir,
    #     labels_dir=labels_dir,
    #     yaml_path=yaml_path,
    #     desired_class_names=desired_class_names,
    #     output_dir=output_dir,
    #     max_workers=max_workersdd
    # )
    # 修改 labels 目录下的所有标签文件，重映射类别 ID
    # remap_all_labels(
    #     labels_root=labels_dir,  # 可以是 labels/ 或 labels/train/
    #     original_yaml_path=yaml_path,
    #     new_class_names=['stop','ok','one','three','four','face','like','dislike','call','hand_heart','person','cat','dog']
    # )

    # 修改1: 定义基础路径
#     base_folder_a = "E:/my_yolo/my_yolo/new_yolo/8_labels"  # 第一个标签文件夹基础路径
#     base_folder_b = "E:/my_yolo/my_yolo/new_yolo/face_labels"  # 第二个标签文件夹基础路径
#     base_output_folder = "E:/my_yolo/my_yolo/new_yolo/labels"  # 输出文件夹基础路径

#     classes_a = ["grabbing", "grip", "holy", "point", "call", "three3", "timeout", "xsign", "hand_heart", "hand_heart2",
#  "little_finger", "middle_finger", "take_picture", "dislike", "first", "four", "like", "mute", "ok", "one",
# "palm", "peace", "peace_inverted", "rock", "stop", "stop_inverted", "three", "three2", "two_up", "two_up_inverted",
# "three_gun", "thumb_index", "thumb_index2", "no_gesture"]
#     classes_b = ['person']
#     final_classes = ['stop','ok','one','three','four','face','like','dislike','call','hand_heart','person','cat','dog']
    
#     # 修改2: 修复循环语法，从 range(['train','test','val']) 改为直接迭代列表
#     for split in ['train', 'test', 'val']:
#         # 修改3: 使用 os.path.join 构造子文件夹路径，确保跨平台兼容
#         folder_a = os.path.join(base_folder_a, split)
#         folder_b = os.path.join(base_folder_b, split)
#         output_folder = os.path.join(base_output_folder, split)
        
#         # 修改4: 调用 merge_yolo_labels 并添加错误检查
#         if os.path.exists(folder_a) or os.path.exists(folder_b):
#             merge_yolo_labels(folder_a, folder_b, output_folder, classes_a, classes_b, final_classes)
#             print(f"{split} 分割标签合并完成，输出到: {output_folder}")
#         else:
#             print(f"警告: {split} 分割的输入文件夹 {folder_a} 和 {folder_b} 均不存在，跳过。")
    
#     # 修改5: 移动最终完成消息到循环外部，显示整体完成状态
#     print(f"所有分割的标签合并完成，输出到: {base_output_folder}")

    # 数据集路径和类名（可替换为任意类名列表）A写入到B
    dataset_a_path = "E:/my_yolo/my_yolo/new_yolo/person_labels"  # 替换为A数据集路径
    dataset_b_path = "E:/my_yolo/my_yolo/new_yolo/labels"  # 替换为B数据集路径
    a_classes = ['person']  # 替换为你的A类名列表
    b_classes = ['stop','ok','one','three','four','face','like','dislike','call','hand_heart','person','cat','dog']  # 替换为你的B类名列表
    
    merge_yolo_labels(dataset_a_path, dataset_b_path, a_classes, b_classes, max_workers=4)
    print("合并完成！")

    # # 数据集根目录
    # dataset_path = "E:/my_yolo/my_yolo/new_yolo"  # 请替换为你的数据集路径
    # filter_yolo_dataset(dataset_path, max_workers=4)
    # print("过滤完成！")


