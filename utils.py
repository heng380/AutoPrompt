"""
工具函数：文件处理、数据加载等
"""
import os
import json
import csv
import pandas as pd
from typing import List, Dict, Any
from werkzeug.utils import secure_filename


ALLOWED_EXTENSIONS = {'csv', 'json', 'jsonl'}


def allowed_file(filename: str) -> bool:
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """
    加载数据集文件
    
    Args:
        file_path: 文件路径
        
    Returns:
        数据集列表
    """
    ext = file_path.rsplit('.', 1)[1].lower()
    
    if ext == 'csv':
        return load_csv(file_path)
    elif ext == 'json':
        return load_json(file_path)
    elif ext == 'jsonl':
        return load_jsonl(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {ext}")


def load_csv(file_path: str) -> List[Dict[str, Any]]:
    """加载 CSV 文件（支持逗号和制表符分隔）"""
    # 尝试自动检测分隔符
    try:
        # 先尝试读取第一行来判断分隔符
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            # 如果第一行包含制表符，使用制表符分隔
            if '\t' in first_line:
                df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
            else:
                df = pd.read_csv(file_path, encoding='utf-8')
    except Exception as e:
        # 如果出错，尝试默认方式
        df = pd.read_csv(file_path, encoding='utf-8')
    
    # 转换为字典列表
    # 假设 CSV 有 'input' 和 'ground_truth' 列
    # 如果没有，尝试自动识别
    data = []
    for _, row in df.iterrows():
        item = {}
        
        # 尝试找到输入列（可能是 'input', 'question', 'text' 等）
        input_cols = ['input', 'question', 'text', 'query', 'prompt', 'headline', 'title', 'content']
        input_col = None
        for col in input_cols:
            if col.lower() in [c.lower() for c in df.columns]:
                # 找到匹配的列（不区分大小写）
                input_col = [c for c in df.columns if c.lower() == col.lower()][0]
                break
        
        if input_col:
            input_value = row[input_col]
            # 处理 NaN 值
            if pd.isna(input_value):
                # 如果输入列为 NaN，尝试使用第二列
                if len(df.columns) > 1:
                    item['input'] = str(row.iloc[1]) if not pd.isna(row.iloc[1]) else ''
                else:
                    item['input'] = ''
            else:
                item['input'] = str(input_value)
        else:
            # 如果没有找到，尝试使用第二列（通常第一列是ID）
            if len(df.columns) > 1:
                item['input'] = str(row.iloc[1]) if not pd.isna(row.iloc[1]) else ''
            else:
                item['input'] = str(row.iloc[0]) if not pd.isna(row.iloc[0]) else ''
        
        # 尝试找到标准答案列
        gt_cols = ['ground_truth', 'answer', 'label', 'output', 'target', 'typography', 'category', 'class']
        gt_col = None
        for col in gt_cols:
            if col.lower() in [c.lower() for c in df.columns]:
                # 找到匹配的列（不区分大小写）
                gt_col = [c for c in df.columns if c.lower() == col.lower()][0]
                break
        
        if gt_col:
            gt_value = row[gt_col]
            # 处理 NaN 值
            if pd.isna(gt_value):
                item['ground_truth'] = ''
            else:
                item['ground_truth'] = str(gt_value)
        else:
            item['ground_truth'] = ''
        
        # 添加其他列
        for col in df.columns:
            if col not in [input_col, gt_col]:
                item[col] = str(row[col])
        
        data.append(item)
    
    return data


def load_json(file_path: str) -> List[Dict[str, Any]]:
    """加载 JSON 文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 如果是单个对象，转换为列表
    if isinstance(data, dict):
        data = [data]
    
    # 确保每个项目都有 'input' 字段
    for item in data:
        if 'input' not in item:
            # 尝试找到输入字段
            for key in ['question', 'text', 'query', 'prompt']:
                if key in item:
                    item['input'] = item[key]
                    break
    
    return data


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """加载 JSONL 文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if 'input' not in item:
                    for key in ['question', 'text', 'query', 'prompt']:
                        if key in item:
                            item['input'] = item[key]
                            break
                data.append(item)
    return data


def save_uploaded_file(file, upload_folder: str) -> str:
    """
    保存上传的文件
    
    Args:
        file: Flask 文件对象
        upload_folder: 上传文件夹路径
        
    Returns:
        保存的文件路径
    """
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, filename)
        file.save(file_path)
        return file_path
    else:
        raise ValueError("不支持的文件类型")

