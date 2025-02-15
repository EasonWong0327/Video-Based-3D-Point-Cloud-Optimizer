import os
import re
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import open3d as o3d
import torch.nn as nn
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF
from MinkowskiEngine import utils as ME_utils
import matplotlib.pyplot as plt
import logging
import argparse
from datetime import datetime
from model.network_TSCM import AustinNet
import shutil
from collections import defaultdict
import subprocess
import pandas as pd
import random
import tempfile
from scipy.spatial import KDTree
from utils import vpcc_catch_new_origin

current_date = datetime.now()
formatted_date = str(current_date.strftime("%Y%m%d"))
task_name = 'Train_TSCM_0215_{}'.format(formatted_date)
model_save_dir = './model_saved/' + task_name
os.makedirs(model_save_dir, exist_ok=True)
log_file = task_name + '.log'
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


# 自定义排序函数
def natural_sort_key(filename):
    # 使用正则表达式分割字符串，提取数字和非数字部分
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', filename)]

def load_ply(file_path):
    """读取 PLY 文件并返回 (N, 6) 的点云数组"""
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    return points

def save_ply_with_open3d(points, save_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(save_path, pcd, write_ascii=True)
    print(f"保存点云到文件 {save_path}")

def _save_ply(points, file_path):
    # 使用open3d保存点云数据为ply格式，不包含颜色
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)  # 只保存坐标信息
    o3d.io.write_point_cloud(file_path, pcd, write_ascii=True)

def merge_blocks(block_folder, output_folder):
    """
    适应情况：文件夹内有多个epoch的block文件进行合并
    """
    blocks = defaultdict(lambda: defaultdict(list))

    # 遍历块文件夹中的所有块文件
    for block_file in sorted(os.listdir(block_folder)):
        if block_file.endswith('.ply'):
            block_file = os.path.basename(block_file)

            parts = block_file.split('_')
            name = '_'.join(parts[:3])
            epoch = str(parts[-1]).replace('epoch','').replace('.ply','')
            block_path = os.path.join(block_folder, block_file)
            blocks[name][epoch].append(block_path)

    # 合并每个特定部分的块
    for name, epoch_files in blocks.items():
        for epoch, block_files in epoch_files.items():
            all_points = []

            for block_file in block_files:
                block_points = load_ply(block_file)
                all_points.append(block_points)

            all_points = np.vstack(all_points)

            output_file = os.path.join(output_folder, f"{name}_eva_merged_epoch{epoch}.ply")

            save_ply_with_open3d(all_points, output_file)
            print(f"合并后的点云保存为: {output_file}")


class PointCloudDataset(Dataset):
    def __init__(self, folder_o, folder_a, folder_b):
        self.folder_O = folder_o
        self.folder_A = folder_a
        self.folder_B = folder_b
        self.file_pairs = self._get_file_pairs()


    def _get_file_pairs(self):
        # 获取folder_A和folder_B中的文件对
        files_O = sorted([f for f in os.listdir(self.folder_O) if f.endswith('.ply')], key=natural_sort_key)
        files_A = sorted([f for f in os.listdir(self.folder_A) if f.endswith('.ply')], key=natural_sort_key)
        files_B = sorted([f for f in os.listdir(self.folder_B) if f.endswith('.ply')], key=natural_sort_key)
        # 随机打乱文件对列表
        file_pairs = list(zip(files_O, files_A, files_B))
        # random.shuffle(file_pairs)
        return file_pairs

    def _load_ply(self, file_path):
        # 使用open3d加载ply文件
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        # print(points.shape, colors.shape)
        if colors.shape[0] > 0 and colors.shape[1] > 0:  # 检查颜色信息是否存在
            return np.hstack((points, colors))
        else:
            return points

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        # 加载已经保存的预处理结果
        file_O, file_A, file_B = self.file_pairs[idx]
        print('Train---PointCloudDataset load:','\n',file_O,'\n', file_A,'\n', file_B, '\n')
        logging.info('Train---PointCloudDataset load:\n%s\n%s\n%s\n', file_O, file_A, file_B)
        adjusted_chunk_O = self._load_ply(os.path.join(self.folder_O, file_O))
        adjusted_chunk_A = self._load_ply(os.path.join(self.folder_A, file_A))
        adjusted_chunk_B = self._load_ply(os.path.join(self.folder_B, file_B))

        # 转换为张量并返回
        adjusted_chunk_O = torch.tensor(adjusted_chunk_O, dtype=torch.float32)
        adjusted_chunk_A = torch.tensor(adjusted_chunk_A, dtype=torch.float32)
        adjusted_chunk_B = torch.tensor(adjusted_chunk_B, dtype=torch.float32)

        return adjusted_chunk_O, adjusted_chunk_A, adjusted_chunk_B


def check_loss_trend(loss_list:list, threshold=0.001):
    # 检查损失列表是否至少有5个元素
    if len(loss_list) < 6:
        return True
    # 获取倒数5轮的损失
    first_five_losses = loss_list[-5:]
    # 计算最大值和最小值
    max_loss = max(first_five_losses)
    min_loss = min(first_five_losses)
    # 检查最大值和最小值的差值是否小于0.001
    return (max_loss - min_loss) > threshold


def position_loss(origin,compress,pred):
    pred_loss = torch.nn.functional.mse_loss(pred, origin)
    c_loss = torch.nn.functional.mse_loss(compress, origin)
    return pred_loss,c_loss


def vpcc_kd_loss(origin,pred,compress,new_origin_c, cache=True):
    '''

    :param origin: 1000,3
    :param pred:   700,3       pred_undup:650,3   new_origin:650,3
    :param compress: 700,3
    :param new_origin_c: 700,3
    :return: pred_loss(参与train计算),c_loss(画图)
    '''
    if not cache:
        # 1.拿predict和origin使用VPCC找new_origin，找到的new_origin作为model的Y
        # new_pred: pc-error对pred进行了某种排序
        new_pred, new_origin_1 = vpcc_catch_new_origin(pred, origin)
        # 2.pred和新找到的new_origin做MSE,作为train的loss
        device = pred.device
        new_origin_1 = new_origin_1.to(device)
        new_pred = new_pred.to(device)
        pred_loss = torch.nn.functional.mse_loss(new_pred,new_origin_1)

        # 3.compress和new_origin_c做MSE,作为画图的loss（尽管2个origin不同，但是权宜之计，也能体现loss的GAP）
        c_loss = torch.nn.functional.mse_loss(compress, new_origin_c)
        return pred_loss, c_loss, new_origin_1
    else:
        device = pred.device
        new_origin_1 = origin.to(device)
        pred_loss = torch.nn.functional.mse_loss(pred, new_origin_1)
        c_loss = torch.nn.functional.mse_loss(compress, new_origin_c)
        return pred_loss, c_loss
    # debug
    # def save_as_ply(tensor, filename, colors=None):
    #     points = tensor.cpu().detach().numpy()
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(points)
    #     if colors is not None:
    #         pcd.colors = o3d.utility.Vector3dVector(colors)
    #     o3d.io.write_point_cloud(filename, pcd, write_ascii=True)
    #     print(f"Saved point cloud to {filename}")
    # save_as_ply(origin, '/home/jupyter-eason/project/upsampling/pc_enhance_simple_script/debug/origin.ply')
    # save_as_ply(pred, '/home/jupyter-eason/project/upsampling/pc_enhance_simple_script/debug/pred.ply')
    # save_as_ply(compress, '/home/jupyter-eason/project/upsampling/pc_enhance_simple_script/debug/compress.ply')
    # save_as_ply(new_origin_c, '/home/jupyter-eason/project/upsampling/pc_enhance_simple_script/debug/new_origin_c.ply')
    # save_as_ply(new_origin_1, '/home/jupyter-eason/project/upsampling/pc_enhance_simple_script/debug/new_origin_1.ply')




def train_model(model, data_loader, optimizer, scheduler, device,
                eva_origin_file_dir,
                eva_new_origin_file_dir,
                eva_test_file_dir,
                eva_predict_output_dir,
                eva_merged_output_dir,
                point_threshold=1000,
                epochs=10,
                blocks_per_epoch=1,
                save_epoch_num=2,
                eva_status=True
                ):
    if os.path.exists(eva_predict_output_dir):
        shutil.rmtree(eva_predict_output_dir)
    if os.path.exists(eva_merged_output_dir):
        shutil.rmtree(eva_merged_output_dir)

    model = model.to(device).float()

    # 用于记录每个epoch的平均损失
    epoch_losses, epoch_compress_losses = [], []
    eva_psnr_results = pd.DataFrame()

    # 用于追踪训练阶段和缓存状态
    training_stage = "stage1"  # 初始为stage1
    cached_origins = {}  # 用字典存储每个block的cached origin
    cache_epoch = None  # 初始时没有缓存
    CACHE_DURATION = 10  # 缓存使用的epoch数量

    for epoch in range(epochs):
        model.train()
        print(f'开始第{epoch}轮次')
        logger.info(f'开始第{epoch}轮次')

        total_loss, compress_loss = 0, 0
        num_batches = 0
        block_buffer_O, block_buffer_A, block_buffer_B = [], [], []

        # 检查是否需要开始新的缓存周期
        if training_stage == "stage2" and (cache_epoch is None or epoch - cache_epoch >= CACHE_DURATION):
            cached_origins.clear()
            cache_epoch = epoch
            logger.info(f'Epoch {epoch}: Starting new cache period in Stage 2')

        for batch_idx, (chunks_O, chunks_A, chunks_B) in enumerate(data_loader):
            if chunks_A.shape[1] > point_threshold:
                block_buffer_O.extend(chunks_O)
                block_buffer_A.extend(chunks_A)
                block_buffer_B.extend(chunks_B)

            if len(block_buffer_A) >= blocks_per_epoch:
                for i in range(0, len(block_buffer_A), blocks_per_epoch):
                    coords_O_batch, coords_A_batch, coords_B_batch = [], [], []

                    for j in range(min(blocks_per_epoch, len(block_buffer_A) - i)):
                        coords_O_batch.append(block_buffer_O[i + j][:, :3])
                        coords_A_batch.append(block_buffer_A[i + j][:, :3])
                        coords_B_batch.append(block_buffer_B[i + j][:, :3])

                    coords_O_batch = np.concatenate(coords_O_batch, axis=0)
                    coords_A_batch = np.concatenate(coords_A_batch, axis=0)
                    coords_B_batch = np.concatenate(coords_B_batch, axis=0)

                    features_O = torch.tensor(coords_O_batch, dtype=torch.float32).to(device)
                    features_A = torch.tensor(coords_A_batch, dtype=torch.float32).to(device)
                    features_B = torch.tensor(coords_B_batch, dtype=torch.float32).to(device)

                    def add_index_column(coords):
                        return torch.cat((torch.zeros(coords.size(0), 1, dtype=torch.float32).to(device), coords),
                                       dim=1)

                    coordinates_O = add_index_column(torch.tensor(coords_O_batch, dtype=torch.float32).to(device))
                    coordinates_A = add_index_column(torch.tensor(coords_A_batch, dtype=torch.float32).to(device))
                    coordinates_B = add_index_column(torch.tensor(coords_B_batch, dtype=torch.float32).to(device))

                    new_origin_c = features_A
                    compress = ME.SparseTensor(features=features_B, coordinates=coordinates_B)
                    output = model(compress)

                    # 检查是否需要从Stage1转换到Stage2
                    if training_stage == "stage1" and not check_loss_trend(epoch_losses):
                        training_stage = "stage2"
                        cache_epoch = epoch  # 重置缓存epoch
                        cached_origins.clear()  # 清空缓存
                        logger.info(f'Epoch {epoch}: Transitioning to Stage 2')

                    if training_stage == "stage1":
                        # Stage 1: 使用position_loss
                        loss, comp_loss = position_loss(
                            origin=new_origin_c,
                            compress=compress.F.float(),
                            pred=output.F.float(),
                        )
                        logger.info(f'Epoch-{epoch} Stage-1 Using position_loss')
                    else:
                        # Stage 2: 使用vpcc_kd_loss，带缓存机制
                        block_key = f"block_{batch_idx}_{i}"

                        if epoch == cache_epoch:  # 缓存周期的第一个epoch
                            # 计算并缓存新的origin points
                            loss, comp_loss, new_origin_1 = vpcc_kd_loss(
                                origin=features_O,
                                pred=output.F.float(),
                                compress=compress.F.float(),
                                new_origin_c=new_origin_c,
                                cache=False
                            )
                            # 存储计算结果
                            cached_origins[block_key] = {
                                'new_origin_1': new_origin_1.detach()
                            }
                            logger.info(f'Epoch-{epoch} Block-{block_key}: Computing and caching new origin points')
                        else:
                            # 使用缓存的origin points
                            cached_data = cached_origins[block_key]
                            loss, comp_loss = vpcc_kd_loss(
                                origin=cached_data['new_origin_1'],
                                pred=output.F.float(),
                                compress=compress.F.float(),
                                new_origin_c=new_origin_c,
                                cache=True
                            )
                            logger.info(f'Epoch-{epoch} Block-{block_key}: Using cached origin points')

                    print('Pred LOSS:', loss.item(),
                          '---Compress LOSS:', comp_loss.item(),
                          '---GAP:', float(comp_loss.item() - loss.item()),
                          '---Stage:', training_stage)
                    logger.info('Pred LOSS: %f --- Compress LOSS: %f --- GAP: %f --- Stage: %s',
                               loss.item(), comp_loss.item(), float(comp_loss.item() - loss.item()), training_stage)

                    total_loss += loss.item()
                    compress_loss += comp_loss.item()
                    num_batches += 1

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                block_buffer_O.clear()
                block_buffer_A.clear()
                block_buffer_B.clear()

        # 计算平均损失
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_compress_loss = compress_loss / num_batches if num_batches > 0 else 0
        logger.info(f"Epoch {epoch + 1}/{epochs}, Average Pred Loss: {avg_loss:.4f}")
        logger.info(f"Epoch {epoch + 1}/{epochs}, Average Compress Loss: {avg_compress_loss:.4f}")
        logger.info(f"Current Training Stage: {training_stage}")

        epoch_losses.append(avg_loss)
        epoch_compress_losses.append(avg_compress_loss)

        # 加入evaluation
        if eva_status:
            eva_res = evaluate_model(model, epoch, device,
                                   origin_block_dir=eva_origin_file_dir,
                                   eva_new_origin_file_dir=eva_new_origin_file_dir,
                                   compress_block_dir=eva_test_file_dir,
                                   predict_block_save_dir=eva_predict_output_dir,
                                   merged_output_dir=eva_merged_output_dir,
                                   )
            eva_psnr_results = pd.concat([eva_psnr_results, eva_res], ignore_index=True)
            eva_psnr_results.to_excel('eva_psnr_results.xlsx')

        # 每第N个epoch保存模型权重
        if (epoch + 1) % save_epoch_num == 0:
            save_path = './model/' + task_name + '/' + str(epoch + 1) + '_model_residual.pth'
            torch.save(model.state_dict(), save_path)
            logger.info(f"Model saved at epoch {epoch + 1} to {save_path}")
        scheduler.step()

        # 绘制损失曲线
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, epoch + 2), epoch_losses, label='Training Loss', color='blue')
        plt.plot(range(1, epoch + 2), epoch_compress_losses, label='Compress Loss', linestyle='--', color='red')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Training Loss vs Epochs (Current Stage: {training_stage})')
        plt.legend()
        plt.savefig(f'My_Net_Residual_Loss_{formatted_date}.png')
        plt.close()

    # 训练完成后的最终损失图
    plt.figure(figsize=(12, 6))
    plt.plot(range(epochs), epoch_losses, label='Training Loss', color='blue')
    plt.plot(range(epochs), epoch_compress_losses, label='Compress Loss', linestyle='--', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss vs Epochs (Final)')
    plt.legend()
    plt.savefig(f'My_Net_Residual_Loss_Last_{formatted_date}.png')
    plt.show()

def predict_on_blocks(model, blocks, device):
    model.eval()
    coords_B = torch.tensor(blocks[:, :3], dtype=torch.float32).to(device)

    # 添加第四维用于批次索引
    batch_index = torch.full((coords_B.shape[0], 1), 0, dtype=torch.int32, device=device)
    int_coords_B = torch.cat([batch_index, coords_B.int()], dim=1)

    # logger.info('Input features: %s', normalized_coords_B)
    # logger.info('Input coordinates: %s', int_coords_B)

    inputs = ME.SparseTensor(features=coords_B, coordinates=int_coords_B)

    with torch.no_grad():  # 禁用梯度计算
        logging.info(f'Training mode:{model.training}')
        output = model(inputs)
    print('model_predict_res:',output.F)
    # logger.info('Model prediction results: %s', output.F)
    # logger.info('Shape check: inputs: %s, output: %s',  inputs.shape, output.shape)

    # 清理未使用的张量
    del coords_B, inputs
    torch.cuda.empty_cache()
    points = output.F.cpu().numpy()
    return points


def psnr(a_file,b_file):
    # MPEG工具路径
    mpeg_tool_path = "/home/jupyter-eason/data/software/mpeg-pcc-tmc2/bin/PccAppMetrics"
    mpeg_tool_path2 = '/home/jupyter-eason/project/upsampling/mpeg-pcc-dmetric-0.13.05/test/pc_error'
    resolution = "1023"
    frame_count = "1"
    print('PSNR:',a_file,b_file)
    # 构建命令
    command = [
        mpeg_tool_path,
        f"--uncompressedDataPath={a_file}",
        f"--reconstructedDataPath={b_file}",
        f"--resolution={resolution}",
        f"--frameCount={frame_count}"
    ]

    # 执行命令并获取输出
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output = result.stdout
    # print(output)
    base_a = os.path.basename(a_file)
    base_b = os.path.basename(b_file)

    # 使用正则表达式提取 mseF,PSNR (p2point) 的值
    match = re.search(r'mseF,PSNR \(p2point\): ([\d.]+)', output)
    if match:
        mseF_psnr_value = match.group(1)  # 提取匹配的值
        print(f"提取的值: {mseF_psnr_value}")
    else:
        print("未找到匹配的值。")
        mseF_psnr_value = 0

    match1 = re.search(r'mse1,PSNR \(p2point\): ([\d.]+)', output)
    if match1:
        mseF_psnr_value1 = match1.group(1)  # 提取匹配的值
        print(f"提取的值: {mseF_psnr_value1}")
    else:
        print("未找到匹配的值。")
        mseF_psnr_value1 = 0

    match2 = re.search(r'mse2,PSNR \(p2point\): ([\d.]+)', output)
    if match2:
        mseF_psnr_value2 = match2.group(1)  # 提取匹配的值
        print(f"提取的值: {mseF_psnr_value2}")
    else:
        print("未找到匹配的值。")
        mseF_psnr_value2 = 0

    match3 = re.search(r'mse1      \(p2point\): ([\d.]+)', output)
    if match3:
        mseF_value = match3.group(1)  # 提取匹配的值
        print(f"提取的值: {mseF_value}")
    else:
        print("未找到匹配的值。")

    print('PSNR:',a_file, b_file, mseF_psnr_value, mseF_value)
    str_res = 'A: '+ base_a + '  TO  B: ' + base_b + ', PSNR: ' + str(mseF_psnr_value1) + '\n' + 'B: '+ base_b + '  TO  A: ' + base_a + ', PSNR: ' + str(mseF_psnr_value2) + '\nFinal PSNR: ' + str(mseF_psnr_value)
    int_res = (mseF_psnr_value1, mseF_psnr_value2)
    return (str_res, int_res, mseF_value)

def psnr_whole_ply(epoch, original_path, compress_path, predict_path):
    '''
    计算整体的PLY PSNR   三个文件夹内的PLY数量应该一致
    :param epoch:
    :param original_path:
    :param compress_path:
    :param predict_path:
    :return:
    '''
    original_files = sorted([ f for f in os.listdir(original_path) if f.endswith('.ply')])
    compress_files = sorted([ f for f in os.listdir(compress_path) if f.endswith('.ply')])
    predict_files  = sorted([ f for f in os.listdir(predict_path) if f.endswith('.ply')])

    original_files = [ os.path.join(original_path,f) for f in original_files]
    compress_files = [ os.path.join(compress_path,f) for f in compress_files]
    predict_files  = [ os.path.join(predict_path,f) for f in predict_files]

    # 1.预测与原图比较
    res_1 = [psnr(a, b)[0] for a, b in zip(predict_files, original_files)]
    # 4.压缩与原图比较
    res_4 = [psnr(a, b)[0] for a, b in zip(compress_files, original_files)]
    data = {
        "epoch": [epoch] * len(res_1),
        "original_vs_predict": res_1,
        "compress_vs_original": res_4
    }
    pd_data = pd.DataFrame(data)
    return pd_data

def psnr_block(epoch, original_files, compress_files, predict_files, keyword='0536'):
    original_files = [x for x in original_files if keyword in x]
    compress_files = [x for x in compress_files if keyword in x]
    predict_files = [x.replace('S26C03R03_rec','soldier_predict') for x in predict_files if keyword in x]

    # 计算psnr
    # 1.预测与原图比较
    res_1 = [psnr(a, b)[1] for a, b in zip(predict_files, original_files)]
    # 4.压缩与原图比较
    res_4 = [psnr(a, b)[1] for a, b in zip(compress_files, original_files)]
    predict_vs_origin = [float(x[0]) for x in res_1]
    origin_vs_predict = [float(x[1]) for x in res_1]

    compress_vs_origin = [float(x[0]) for x in res_4]
    origin_vs_compress = [float(x[1]) for x in res_4]

    block_index = list(range(len(predict_vs_origin)))

    # 创建图形
    plt.figure(figsize=(30, 6))
    # 绘制四个列表
    plt.plot(block_index, predict_vs_origin, label='Predict vs Origin', marker='o')
    plt.plot(block_index, origin_vs_predict, label='Origin vs Predict', marker='x')
    plt.plot(block_index, compress_vs_origin, label='Compress vs Origin', marker='s')
    plt.plot(block_index, origin_vs_compress, label='Origin vs Compress', marker='d')

    # 设置标题和标签
    plt.title(f'block_psnr_epoch_{epoch}')
    plt.xlabel('Block Index')
    plt.ylabel('PSNR')
    plt.legend()
    plt.grid()

    # 保存图形到文件夹
    png_dir = '/home/jupyter-eason/project/upsampling/pc_enhance_simple_script/png/'
    output_filename = png_dir + f'Block_psnr_epoch_{epoch}.png'
    plt.savefig(output_filename)
    plt.close()


def mse(file1,file2):
    print('MSE:',file1,file2)
    points1 = load_ply(file1)
    points2 = load_ply(file2)
    return float(np.mean((points1 - points2) ** 2))

def mse_block(epoch, original_files, compress_files, predict_files, keyword='0536'):
    original_files = [x for x in original_files if keyword in x]
    compress_files = [x for x in compress_files if keyword in x]
    predict_files = [x.replace('S26C03R03_rec','soldier_predict') for x in predict_files if keyword in x]

    # 1.预测与原图比较
    predict_vs_origin = [mse(a, b) for a, b in zip(predict_files, original_files)]

    # 4.压缩与原图比较
    compress_vs_origin = [mse(a, b) for a, b in zip(compress_files, original_files)]

    block_index = list(range(len(predict_vs_origin)))

    # 创建图形
    plt.figure(figsize=(30, 6))
    # 绘制四个列表
    plt.plot(block_index, predict_vs_origin, label='Predict vs Origin', marker='o')
    plt.plot(block_index, compress_vs_origin, label='Compress vs Origin', marker='s')

    # 设置标题和标签
    plt.title(f'block_mse_epoch_{epoch}')
    plt.xlabel('Block Index')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid()
    # 保存图形到文件夹
    png_dir = '/home/jupyter-eason/project/upsampling/pc_enhance_simple_script/png/'
    output_filename = png_dir + f'Block_MSE_epoch_{epoch}.png'
    plt.savefig(output_filename)
    plt.close()

def evaluate_model(model, epoch, device,
                   origin_block_dir,
                    eva_new_origin_file_dir,
                   compress_block_dir,
                   predict_block_save_dir,
                   merged_output_dir
                   ):

    """
    :param origin_block_dir:
    :param merged_output_dir:
    :param predict_block_save_dir:
    :param compress_block_dir: './test/test2/soldier/block100/block_compress'
    :param device:
    :return:
    """

    # 保存预测的block   创建输出文件夹 './test/test2/soldier/evaluate_block/epoch_0'
    predict_block_save_dir = os.path.join(predict_block_save_dir, f'epoch_{epoch}')
    os.makedirs(predict_block_save_dir, exist_ok=True)
    # 保存合并的   创建输出文件夹 './test/test2/soldier/evaluate_merged/epoch_0'
    merged_output_dir = os.path.join(merged_output_dir, f'epoch_{epoch}')
    os.makedirs(merged_output_dir, exist_ok=True)

    # 新建一个文件夹，供计算block PSRN，内容和predict_block_save_dir一致，只是ply名称变了
    block_psnr_dir = str(predict_block_save_dir).replace('/epoch_', '_rename/epoch_')
    os.makedirs(block_psnr_dir, exist_ok=True)


    # 获取 compress_dir 中的所有 .ply 文件
    files_B = sorted([os.path.join(compress_block_dir, f) for f in os.listdir(compress_block_dir) if f.endswith('.ply')])
    # 获取 origin 中的所有 .ply 文件
    files_A = sorted([os.path.join(origin_block_dir, f) for f in os.listdir(origin_block_dir) if f.endswith('.ply')])

    # 获取 new_origin 中的所有 .ply 文件
    files_new_A = sorted([os.path.join(eva_new_origin_file_dir, f) for f in os.listdir(eva_new_origin_file_dir) if f.endswith('.ply')])

    for file_B in files_B:
        # 提取文件名部分，用于输出文件的命名
        base_name_B = os.path.basename(file_B)
        # 加载并切分文件 B
        points_B = load_ply(file_B)
        # 预测并合并所有块
        predicted_points = predict_on_blocks(model, points_B, device)

        # 保存合并后的点云
        base_name_B_out = base_name_B.replace('.ply','')
        base_name_B_rename = base_name_B.replace('S26C03R03_rec','soldier_predict')

        output_file_path = os.path.join(predict_block_save_dir, f"{base_name_B_out}_EvaPredicted_epoch_{str(epoch)}.ply")
        output_file_path_rename = os.path.join(block_psnr_dir, base_name_B_rename)

        save_ply_with_open3d(predicted_points, output_file_path)
        save_ply_with_open3d(predicted_points, output_file_path_rename)

    # 获取 predict 中的所有 .ply 文件  这里得用block_psnr_dir文件夹
    files_p = sorted([os.path.join(block_psnr_dir, f) for f in os.listdir(block_psnr_dir) if f.endswith('.ply')])

    # 计算block-PSNR,默认生成0536的图
    psnr_block(epoch, files_A, files_B, files_p, keyword='0536')
    # 计算mse-PSNR,默认生成0536的图
    mse_block(epoch,files_new_A, files_B, files_p,  keyword='0536')

    merge_blocks(block_folder=predict_block_save_dir, output_folder=merged_output_dir)

    # 此时有3种ply文件
    # 1.original
    # 2.compress （VPCC压缩后）
    # 3.predict (每个epoch都会预测出来2张，主要是观察PSNR与compress-original的PSNR，能与baseline-loss和real-loss对应上)

    # './test/test2/soldier
    base_path = compress_block_dir.split('block')[0]

    # ./test/test2/soldier/original
    original_path = os.path.join(base_path, 'original')
    compress_path = os.path.join(base_path, 'compress')
    # './test/test2/soldier/evaluate_merged/epoch_0'
    predict_path = merged_output_dir
    # 计算整体ply的PSNR
    pd_psnr = psnr_whole_ply(epoch, original_path, compress_path, predict_path)
    return pd_psnr


def main(mode):
    if mode == 'full_run':
        folder_O = './data30/soldier/block100/block_origin'
        # 未去重的  new_compress无重复 new_origin有重复
        folder_A = './data30/soldier/block100/new_original'
        folder_B = './data30/soldier/block100/block_compress'
    else:
        raise ValueError("Invalid mode. Use 'test' or 'full'.")
    # 每个epoch结束后，拿这个数据集进行评估，得到PSNR-BLOCK MSE-BLOCK 总体PLY的双向PSNR值
    # 测试数据集也有3类
    # 1.原始compress切块
    eva_test_file_dir = './test/test2/soldier/block100/block_compress'
    # 2.原始origin切块
    eva_origin_file_dir = './test/test2/soldier/block100/block_origin'
    # 3.VPCC：compress从origin里面找到的（为了参与mse计算-需要shape一致）
    eva_new_origin_file_dir = './test/test2/soldier/block100/block_new_origin'

    # 保存预测的block数据，里面有epoch0 1 2 3等  每次执行会删掉重新建立
    eva_predict_output_dir = './test/test2/soldier/evaluate_block'
    # 保存合并的  每次执行会删掉重新建立
    eva_merged_output_dir = './test/test2/soldier/evaluate_merged'

    dataset = PointCloudDataset(
        folder_o=folder_O,
        folder_a=folder_A,
        folder_b=folder_B
                                )

    data_loader = DataLoader(dataset, batch_size=1)

    model = AustinNet()

    # def weights_init(m):
    #     if isinstance(m, ME.MinkowskiConvolution):
    #         # 使用 He 初始化
    #         nn.init.kaiming_normal_(m.kernel, mode='fan_out', nonlinearity='relu')  # 使用 m.kernel 代替 m.weight
    #         if m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, ME.MinkowskiBatchNorm):
    #         # 对于 BatchNorm 层，通常初始化权重为 1，偏置为 0
    #         if hasattr(m, 'alpha') and m.alpha is not None:
    #             nn.init.constant_(m.alpha, 1)  # 使用 m.alpha 代替 m.weight
    #         if hasattr(m, 'beta') and m.beta is not None:
    #             nn.init.constant_(m.beta, 0)  # 使用 m.beta 代替 m.biasias
    #
    # model.apply(weights_init)

    optimizer = torch.optim.Adam([{"params": model.parameters(), 'lr': 0.001}],
                                 betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_model(model = model,
                data_loader = data_loader,
                optimizer = optimizer,
                scheduler = scheduler,
                device=torch.device("cuda:0"),
                eva_origin_file_dir = eva_origin_file_dir,
                eva_new_origin_file_dir = eva_new_origin_file_dir,
                eva_test_file_dir=eva_test_file_dir,
                eva_predict_output_dir=eva_predict_output_dir,
                eva_merged_output_dir=eva_merged_output_dir,
                point_threshold=1000,
                epochs=100,
                blocks_per_epoch=1,
                eva_status=False
                )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run training or testing mode.")
    parser.add_argument("mode", choices=["test_process", "full_process", "test_run", "full_run"],
                        help="Mode to run: 'test' or 'full'")
    args = parser.parse_args()

    main(args.mode)
