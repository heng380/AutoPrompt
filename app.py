"""
Flask 主应用：提供 Web 界面
"""
import os
import json
import uuid
import threading
import time
from flask import Flask, render_template, request, jsonify, session, Response, stream_with_context
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from graph import AutoPromptGraph
from utils.logger import get_log_collector, QueueListener
# 直接导入 utils.py 文件（使用 importlib 避免与 utils 目录冲突）
import importlib.util
import os

# 获取 utils.py 文件的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
utils_file_path = os.path.join(current_dir, 'utils.py')

# 加载 utils.py 模块
spec = importlib.util.spec_from_file_location("file_utils", utils_file_path)
file_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(file_utils)

# 从文件模块获取函数
load_dataset = file_utils.load_dataset
save_uploaded_file = file_utils.save_uploaded_file
allowed_file = file_utils.allowed_file

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


# 存储当前优化任务的会话ID和结果
optimization_sessions = {}
session_lock = threading.Lock()

@app.route('/optimize', methods=['POST'])
def optimize_prompt():
    """开始优化 prompt（异步执行）"""
    data = request.get_json()
    
    original_prompt = data.get('prompt', '').strip()
    max_iterations = int(data.get('max_iterations', 3))
    verification_threshold = float(data.get('verification_threshold', 0.5))  # 默认 50%
    candidate_count = int(data.get('candidate_count', 3))  # 默认 3
    
    # 验证阈值范围
    if verification_threshold < 0 or verification_threshold > 1:
        return jsonify({'error': '验证阈值必须在 0-1 之间（0-100%）'}), 400
    
    # 验证候选数量范围
    if candidate_count < 1 or candidate_count > 10:
        return jsonify({'error': '候选数量必须在 1-10 之间'}), 400
    
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
        
        # 生成会话ID和实验ID
        session_id = str(uuid.uuid4())
        experiment_id = str(uuid.uuid4())
        
        # 在新线程中运行优化
        def run_optimization():
            try:
                log_collector = get_log_collector()
                log_collector.clear()
                
                # 创建并运行优化工作流
                graph = AutoPromptGraph(experiment_id=experiment_id)
                result = graph.run(
                    original_prompt=original_prompt,
                    dataset=dataset,
                    max_iterations=max_iterations,
                    experiment_id=experiment_id,
                    verification_threshold=verification_threshold,
                    candidate_count=candidate_count
                )
                
                # 计算最终准确率（使用 graph.run() 返回的准确率，或重新计算）
                final_results = result['final_results']
                correct_count = sum(1 for r in final_results if r.get('is_correct', False))
                accuracy = result.get('final_accuracy', correct_count / len(final_results) * 100 if final_results else 0) / 100  # 转换为 0-1 范围
                
                with session_lock:
                    optimization_sessions[session_id] = {
                        'success': True,
                        'experiment_id': experiment_id,
                        'final_prompt': result['final_prompt'],
                        'final_prompt_source': result.get('final_prompt_source', '最终轮次'),  # prompt 来源：最终轮次 / 历史最高准确率
                        'final_prompt_iteration': result.get('final_prompt_iteration', result['iterations']),  # prompt 对应的轮次
                        'original_prompt': original_prompt,
                        'accuracy': accuracy,
                        'final_accuracy': result.get('final_accuracy', accuracy * 100),  # 最终准确率（百分比）
                        'best_accuracy': result.get('best_accuracy', accuracy * 100),  # 历史最高准确率（百分比）
                        'best_iteration': result.get('best_iteration', result['iterations']),  # 历史最高准确率对应的轮次
                        'correct_count': correct_count,
                        'total_count': len(final_results),
                        'iterations': result['iterations'],
                        'history': result['history'],
                        'final_results': final_results,
                        'all_iteration_results': result.get('all_iteration_results', []),
                        'memory_experiences': result.get('memory_experiences', ''),
                        'completed': True
                    }
                    log_collector.log("优化任务完成！", level="success")
            except Exception as e:
                with session_lock:
                    optimization_sessions[session_id] = {
                        'success': False,
                        'error': str(e),
                        'completed': True
                    }
                    log_collector = get_log_collector()
                    log_collector.log(f"优化过程出错: {str(e)}", level="error")
        
        # 启动优化线程
        thread = threading.Thread(target=run_optimization)
        thread.daemon = True
        thread.start()
        
        # 初始化会话状态
        with session_lock:
            optimization_sessions[session_id] = {
                'completed': False,
                'success': None
            }
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': '优化任务已启动'
        })
    except Exception as e:
        return jsonify({'error': f'启动优化任务出错: {str(e)}'}), 500


@app.route('/logs/<session_id>')
def stream_logs(session_id):
    """SSE 日志流"""
    def generate():
        log_collector = get_log_collector()
        listener = QueueListener()
        log_collector.add_listener(listener)
        
        try:
            # 先发送已存在的日志
            for log_entry in log_collector.get_logs():
                yield f"data: {json.dumps(log_entry)}\n\n"
            
            # 然后实时发送新日志
            while True:
                log_entry = listener.get(timeout=1)
                if log_entry:
                    yield f"data: {json.dumps(log_entry)}\n\n"
                else:
                    # 检查是否完成
                    with session_lock:
                        session_data = optimization_sessions.get(session_id)
                        if session_data and session_data.get('completed'):
                            yield f"data: {json.dumps({'type': 'completed'})}\n\n"
                            break
                    yield f": keepalive\n\n"  # SSE keepalive
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e), 'level': 'error'})}\n\n"
        finally:
            log_collector.remove_listener(listener)
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')


@app.route('/result/<session_id>')
def get_result(session_id):
    """获取优化结果"""
    with session_lock:
        session_data = optimization_sessions.get(session_id)
        if not session_data:
            return jsonify({'error': '会话不存在'}), 404
        
        if not session_data.get('completed'):
            return jsonify({'completed': False}), 200
        
        if session_data.get('success'):
            return jsonify(session_data)
        else:
            return jsonify({'error': session_data.get('error', '未知错误')}), 500


@app.route('/clear', methods=['POST'])
def clear_session():
    """清除 session"""
    session.clear()
    return jsonify({'success': True})


@app.route('/default_data', methods=['GET'])
def get_default_data():
    """获取默认数据集和 prompt"""
    try:
        # 获取项目根目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 读取默认 prompt
        prompt_file = os.path.join(current_dir, 'example_data', 'example_prompt.txt')
        default_prompt = ""
        if os.path.exists(prompt_file):
            with open(prompt_file, 'r', encoding='utf-8') as f:
                default_prompt = f.read().strip()
        
        # 加载默认数据集
        dataset_file = os.path.join(current_dir, 'example_data', 'example_dataset.csv')
        
        if os.path.exists(dataset_file):
            # 保存默认数据集到 session
            dataset = load_dataset(dataset_file)
            session['dataset_path'] = dataset_file
            session['dataset_size'] = len(dataset)
            
            return jsonify({
                'success': True,
                'prompt': default_prompt,
                'dataset': {
                    'filename': 'example_dataset.csv',
                    'size': len(dataset),
                    'preview': dataset[:5] if len(dataset) > 5 else dataset
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': f'默认数据集文件不存在: {dataset_file}'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


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

