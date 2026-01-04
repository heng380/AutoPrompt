"""
Flask 主应用：提供 Web 界面
"""
import os
import json
import uuid
from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from graph import AutoPromptGraph
from utils import load_dataset, save_uploaded_file, allowed_file

# 加载环境变量
load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# 确保上传文件夹存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """上传数据集文件"""
    if 'file' not in request.files:
        return jsonify({'error': '没有文件被上传'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': '不支持的文件格式，请上传 CSV、JSON 或 JSONL 文件'}), 400
    
    try:
        file_path = save_uploaded_file(file, app.config['UPLOAD_FOLDER'])
        
        # 加载数据集预览（前5条）
        dataset = load_dataset(file_path)
        preview = dataset[:5] if len(dataset) > 5 else dataset
        
        # 保存文件路径到 session
        session['dataset_path'] = file_path
        session['dataset_size'] = len(dataset)
        
        return jsonify({
            'success': True,
            'filename': file.filename,
            'size': len(dataset),
            'preview': preview
        })
    except Exception as e:
        return jsonify({'error': f'文件处理错误: {str(e)}'}), 500


@app.route('/optimize', methods=['POST'])
def optimize_prompt():
    """开始优化 prompt"""
    data = request.get_json()
    
    original_prompt = data.get('prompt', '').strip()
    max_iterations = int(data.get('max_iterations', 5))
    
    if not original_prompt:
        return jsonify({'error': 'Prompt 不能为空'}), 400
    
    if 'dataset_path' not in session:
        return jsonify({'error': '请先上传数据集'}), 400
    
    dataset_path = session['dataset_path']
    
    try:
        # 加载数据集
        dataset = load_dataset(dataset_path)
        
        if len(dataset) == 0:
            return jsonify({'error': '数据集为空'}), 400
        
        # 生成实验ID
        experiment_id = str(uuid.uuid4())
        
        # 创建并运行优化工作流
        graph = AutoPromptGraph(experiment_id=experiment_id)
        result = graph.run(
            original_prompt=original_prompt,
            dataset=dataset,
            max_iterations=max_iterations,
            experiment_id=experiment_id
        )
        
        # 计算最终准确率
        final_results = result['final_results']
        correct_count = sum(1 for r in final_results if r.get('is_correct', False))
        accuracy = correct_count / len(final_results) if final_results else 0
        
        return jsonify({
            'success': True,
            'experiment_id': experiment_id,  # 返回实验ID
            'final_prompt': result['final_prompt'],
            'original_prompt': original_prompt,
            'accuracy': accuracy,
            'correct_count': correct_count,
            'total_count': len(final_results),
            'iterations': result['iterations'],
            'history': result['history'],
            'final_results': final_results,  # 返回所有最终结果
            'all_iteration_results': result.get('all_iteration_results', []),  # 包含所有轮次的详细结果
            'memory_experiences': result.get('memory_experiences', '')  # 包含累积的经验
        })
    except Exception as e:
        return jsonify({'error': f'优化过程出错: {str(e)}'}), 500


@app.route('/clear', methods=['POST'])
def clear_session():
    """清除 session"""
    session.clear()
    return jsonify({'success': True})


if __name__ == '__main__':
    # 检查 API Key 配置
    openai_key = os.getenv('OPENAI_API_KEY')
    azure_key = os.getenv('AZURE_OPENAI_API_KEY')
    azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    
    if not openai_key and not (azure_key and azure_endpoint):
        print("警告: 未配置 API Key")
        print("请选择以下方式之一配置：")
        print("1. OpenAI: 设置 OPENAI_API_KEY")
        print("2. Azure OpenAI: 设置 AZURE_OPENAI_API_KEY 和 AZURE_OPENAI_ENDPOINT")
        print("请在 .env 文件中设置，或使用环境变量")
    elif azure_key and azure_endpoint:
        print("✓ 使用 Azure OpenAI")
    else:
        print("✓ 使用 OpenAI")
    
    app.run(debug=True, host='0.0.0.0', port=5001)

