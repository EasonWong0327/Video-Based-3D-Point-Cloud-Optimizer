import torch
import os
import subprocess
import shutil
import open3d as o3d
import time
import tempfile
import numpy as np


def split_list_at_second_occurrence(input_list, value):
    # 记录值出现的次数
    count = 0
    split_index = -1
    # 遍历列表，找到第二个出现的位置
    for i, (item, v) in enumerate(input_list):
        if item == value:
            count += 1
            if count == 2:  # 找到第二个出现
                split_index = i
                break
    # 如果找到了第二个出现的位置，则进行分割
    if split_index != -1:
        return input_list[:split_index], input_list[split_index:]
    else:
        return input_list, []  # 如果没有找到，返回原列表和空列表


def count_points_in_ply(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    num_points = len(pcd.points)
    return num_points


def load_ply(file_path):
    """读取 PLY 文件并返回 (N, 6) 的点云数组"""
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    points_tensor = torch.tensor(points, dtype=torch.float32)
    return points_tensor


def extract_points(output):
    """Extract both A and B points from the command output with their indices"""
    point_pairs = {}  # 使用字典存储，键为A点的索引
    capture_points = False

    for line in output.splitlines():
        if '1. Use infile1 (A) as reference' in line:
            capture_points = True
        elif '2. Use infile2 (B) as reference' in line:
            break

        if capture_points and 'Point A[' in line:
            try:
                # 提取A点的索引
                a_index = int(line.split('A[')[1].split(']')[0])

                # 提取坐标
                parts = line.split(' -> ')
                if len(parts) == 2:
                    # 提取A坐标
                    a_part = parts[0].split('(')[1].split(')')[0]
                    ax, ay, az = map(float, a_part.split(','))

                    # 提取B坐标
                    b_part = parts[1].split('(')[1].split(')')[0]
                    bx, by, bz = map(float, b_part.split(','))

                    point_pairs[a_index] = ((ax, ay, az), (bx, by, bz))
            except Exception:
                continue

    return point_pairs


def vpcc_catch_new_origin(predict, origin):
    def save_as_ply(filename, data):
        with open(filename, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {data.shape[0]}\n")
            f.write("property double x\n")
            f.write("property double y\n")
            f.write("property double z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            for point in data:
                f.write(f"{point[0]} {point[1]} {point[2]} 0 0 0\n")

    # 保存为 PLY 文件
    temp_dir = '/home/jupyter-eason/project/upsampling/pc_enhance_simple_script/tmp/vpcc_tmp'
    origin_file = os.path.join(temp_dir, 'origin.ply')
    pred_file = os.path.join(temp_dir, 'pred.ply')
    save_as_ply(origin_file, origin)
    save_as_ply(pred_file, predict)

    # A TO B   A:pred_file  B:origin_file
    uncompressed_data_path = pred_file
    reconstructed_data_path = origin_file
    print('\n VPCC Catch Check \n')
    print('A:', count_points_in_ply(uncompressed_data_path), 'B:', count_points_in_ply(reconstructed_data_path))
    print('\n')
    # 定义要执行的命令
    command = [
        "/home/jupyter-eason/project/upsampling/mpeg-pcc-dmetric-0.13.05/test/pc_error",
        f"--fileA={uncompressed_data_path}",
        f"--fileB={reconstructed_data_path}",
        "--resolution=1023",
        "--color=0",
        "--dropdups=0",
        "--singlePass=1"
    ]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    output = result.stdout

    # 提取点对，现在返回的是字典，键为A点的索引
    point_pairs = extract_points(output)
    if len(point_pairs) == 0:
        return None, None

    # 获取最大索引值，用于确定点云大小
    max_index = max(point_pairs.keys())

    # 写入PLY文件
    new_origin_file = os.path.join(temp_dir, 'new_origin.ply')
    pred_points_file = os.path.join(temp_dir, 'pred.ply')

    # Write Point A PLY file
    with open(pred_points_file, 'w', encoding='utf-8') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {max_index + 1}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        # 按索引顺序写入点
        for i in range(max_index + 1):
            if i in point_pairs:
                a_point = point_pairs[i][0]
                f.write(f"{a_point[0]} {a_point[1]} {a_point[2]} 255 255 255\n")
            else:
                # 对于没有对应点的索引位置，写入默认值或者0
                f.write("0 0 0 255 255 255\n")

    # Write Point B PLY file (new_origin)
    with open(new_origin_file, 'w', encoding='utf-8') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {max_index + 1}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        # 按索引顺序写入点
        for i in range(max_index + 1):
            if i in point_pairs:
                b_point = point_pairs[i][1]
                f.write(f"{b_point[0]} {b_point[1]} {b_point[2]} 255 255 255\n")
            else:
                # 对于没有对应点的索引位置，写入默认值或者0
                f.write("0 0 0 255 255 255\n")

    return load_ply(pred_points_file).requires_grad_(), load_ply(new_origin_file).requires_grad_()