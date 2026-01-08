// 页面加载时初始化默认数据
document.addEventListener('DOMContentLoaded', function() {
    loadDefaultData();
});

// 加载默认数据集和 prompt
function loadDefaultData() {
    fetch('/default_data')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // 设置默认 prompt
                const promptInput = document.getElementById('promptInput');
                if (promptInput && data.prompt) {
                    promptInput.value = data.prompt;
                }
                
                // 显示默认数据集信息
                if (data.dataset) {
                    document.getElementById('fileName').textContent = data.dataset.filename;
                    document.getElementById('fileSize').textContent = data.dataset.size;
                    uploadPlaceholder.style.display = 'none';
                    uploadInfo.style.display = 'block';
                }
            } else {
                console.warn('加载默认数据失败:', data.error);
            }
        })
        .catch(error => {
            console.error('加载默认数据出错:', error);
        });
}

// 文件上传处理
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadPlaceholder = document.getElementById('uploadPlaceholder');
const uploadInfo = document.getElementById('uploadInfo');

uploadArea.addEventListener('click', () => {
    fileInput.click();
});

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.style.background = '#f0f0ff';
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.style.background = '';
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.style.background = '';
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileUpload(files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileUpload(e.target.files[0]);
    }
});

function handleFileUpload(file) {
    const formData = new FormData();
    formData.append('file', file);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert('上传失败: ' + data.error);
            return;
        }

        document.getElementById('fileName').textContent = data.filename;
        document.getElementById('fileSize').textContent = data.size;
        uploadPlaceholder.style.display = 'none';
        uploadInfo.style.display = 'block';
    })
    .catch(error => {
        console.error('Error:', error);
        alert('上传失败，请重试');
    });
}

function clearUpload() {
    fileInput.value = '';
    uploadPlaceholder.style.display = 'flex';
    uploadInfo.style.display = 'none';
    
    fetch('/clear', {
        method: 'POST'
    });
}

// 日志相关变量
let currentSessionId = null;
let eventSource = null;
let logCheckInterval = null;

function addLogEntry(logEntry) {
    const logContent = document.getElementById('logContent');
    if (!logContent) return;
    
    const logLine = document.createElement('div');
    logLine.className = `log-line log-${logEntry.level || 'info'}`;
    
    // 解析并高亮日志内容
    let message = escapeHtml(logEntry.message || '');
    
    // 检测是否是迭代分隔线
    const isIterationSeparator = /─{10,}.*迭代.*─{10,}/.test(message);
    
    if (isIterationSeparator) {
        // 迭代分隔线样式
        logLine.className = 'log-line log-iteration-separator';
        // 提取迭代号并高亮
        message = message.replace(/(迭代 \d+)/g, '<span class="log-iteration-number">$1</span>');
        logLine.innerHTML = `<span class="log-message">${message}</span>`;
    } else {
        // 高亮 [步骤名称] 部分
        message = message.replace(/\[([^\]]+)\]/g, '<span class="log-step">[$1]</span>');
        
        // 高亮 ✓ 和 ✗
        message = message.replace(/✓/g, '<span class="log-success">✓</span>');
        message = message.replace(/✗/g, '<span class="log-error">✗</span>');
        
        // 高亮数字和百分比
        message = message.replace(/(\d+(?:\.\d+)?%)/g, '<span class="log-number">$1</span>');
        message = message.replace(/(\d+\/\d+)/g, '<span class="log-number">$1</span>');
        
        logLine.innerHTML = `
            <span class="log-timestamp">[${logEntry.timestamp || ''}]</span>
            <span class="log-message">${message}</span>
        `;
    }
    
    logContent.appendChild(logLine);
    
    // 自动滚动到底部（使用多种方式确保滚动生效）
    const scrollToBottom = () => {
        const maxScroll = logContent.scrollHeight - logContent.clientHeight;
        logContent.scrollTop = maxScroll > 0 ? maxScroll : 0;
    };
    
    // 立即尝试滚动
    scrollToBottom();
    
    // 使用 requestAnimationFrame 确保 DOM 更新后再滚动
    requestAnimationFrame(() => {
        scrollToBottom();
    });
    
    // 使用 setTimeout 作为备用，确保内容渲染完成
    setTimeout(scrollToBottom, 0);
}

function startLogStream(sessionId) {
    // 清空日志窗口
    const logContent = document.getElementById('logContent');
    if (logContent) {
        logContent.innerHTML = '';
    }
    
    // 显示日志窗口
    const logWindow = document.getElementById('logWindow');
    if (logWindow) {
        logWindow.style.display = 'block';
    }
    
    // 关闭之前的连接
    if (eventSource) {
        eventSource.close();
    }
    
    // 创建新的 SSE 连接
    eventSource = new EventSource(`/logs/${sessionId}`);
    
    eventSource.onmessage = function(event) {
        try {
            const data = JSON.parse(event.data);
            if (data.type === 'completed') {
                eventSource.close();
                checkOptimizationResult(sessionId);
            } else {
                addLogEntry(data);
            }
        } catch (e) {
            console.error('Error parsing log data:', e);
        }
    };
    
    eventSource.onerror = function(event) {
        console.error('SSE error:', event);
        // 如果连接关闭，开始轮询结果
        if (eventSource.readyState === EventSource.CLOSED) {
            checkOptimizationResult(sessionId);
        }
    };
    
    // 同时设置轮询作为备用
    logCheckInterval = setInterval(() => {
        checkOptimizationResult(sessionId);
    }, 2000);
}

function checkOptimizationResult(sessionId) {
    fetch(`/result/${sessionId}`)
        .then(response => response.json())
        .then(data => {
            if (data.completed === false) {
                // 还在进行中，继续等待
                return;
            }
            
            // 完成，停止轮询
            if (logCheckInterval) {
                clearInterval(logCheckInterval);
                logCheckInterval = null;
            }
            
            if (eventSource) {
                eventSource.close();
                eventSource = null;
            }
            
            const optimizeBtn = document.getElementById('optimizeBtn');
            optimizeBtn.disabled = false;
            optimizeBtn.textContent = '开始优化';
            
            if (data.error) {
                const resultArea = document.getElementById('resultArea');
                resultArea.innerHTML = `
                    <div class="error-message">
                        <strong>错误:</strong> ${data.error}
                    </div>
                `;
                addLogEntry({ message: `错误: ${data.error}`, level: 'error', timestamp: new Date().toLocaleTimeString() });
                return;
            }
            
            if (data.success) {
                displayResults(data);
                addLogEntry({ message: '✅ 优化任务完成！', level: 'success', timestamp: new Date().toLocaleTimeString() });
            }
        })
        .catch(error => {
            console.error('Error checking result:', error);
        });
}

// 开始优化
function startOptimization() {
    const prompt = document.getElementById('promptInput').value.trim();
    const maxIterations = parseInt(document.getElementById('maxIterations').value);

    if (!prompt) {
        alert('请输入 prompt');
        return;
    }

    const optimizeBtn = document.getElementById('optimizeBtn');
    optimizeBtn.disabled = true;
    optimizeBtn.textContent = '优化中...';

    const resultArea = document.getElementById('resultArea');
    resultArea.innerHTML = `
        <div class="loading">
            <div class="spinner"></div>
            <p>正在优化 Prompt，请稍候...</p>
            <p style="color: #999; font-size: 0.9em; margin-top: 10px;">
                这可能需要几分钟时间，请耐心等待
            </p>
            <p style="color: #667eea; font-size: 0.9em; margin-top: 10px;">
                💡 请查看下方的实时日志了解优化进度
            </p>
        </div>
    `;

    fetch('/optimize', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            prompt: prompt,
            max_iterations: maxIterations
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            optimizeBtn.disabled = false;
            optimizeBtn.textContent = '开始优化';
            resultArea.innerHTML = `
                <div class="error-message">
                    <strong>错误:</strong> ${data.error}
                </div>
            `;
            return;
        }
        
        if (data.session_id) {
            currentSessionId = data.session_id;
            startLogStream(data.session_id);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        optimizeBtn.disabled = false;
        optimizeBtn.textContent = '开始优化';
        resultArea.innerHTML = `
            <div class="error-message">
                <strong>错误:</strong> 优化过程出错，请重试
            </div>
        `;
    });
}

let accuracyChart = null;
let currentHistoryData = null;

function displayResults(data) {
    const accuracy = (data.accuracy * 100).toFixed(2);
    currentHistoryData = data.history || [];
    
    // 准备折线图数据
    const chartData = prepareChartData(data.history || []);
    
    const resultArea = document.getElementById('resultArea');
    resultArea.innerHTML = `
        <div class="result-content active">
            <!-- 成功消息 -->
            <div class="card">
                <div class="success-message">
                    ✅ 优化完成！共进行了 ${data.iterations} 轮迭代
                </div>
            </div>

            <!-- 指标卡片 -->
            <div class="card">
                <div class="metrics">
                    <div class="metric-card">
                        <div class="metric-value">${accuracy}%</div>
                        <div class="metric-label">最终准确率</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${data.correct_count}/${data.total_count}</div>
                        <div class="metric-label">正确预测</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${data.iterations}</div>
                        <div class="metric-label">迭代轮次</div>
                    </div>
                </div>
            </div>

            <!-- 准确率折线图 -->
            <div class="card">
                <h2>准确率变化趋势</h2>
                <canvas id="accuracyChart" style="max-height: 400px;"></canvas>
            </div>

            <!-- Prompt 对比（左右排列） -->
            <div class="card">
                <div class="prompt-comparison">
                    <div class="prompt-item">
                        <h3>原始 Prompt</h3>
                        <div class="prompt-box">${escapeHtml(data.original_prompt)}</div>
                    </div>
                    <div class="prompt-item">
                        <h3>优化后的 Prompt</h3>
                        <div class="prompt-box">${escapeHtml(data.final_prompt)}</div>
                    </div>
                </div>
            </div>

            <!-- 优化经验（Memory） -->
            ${data.memory_experiences && data.memory_experiences.trim() ? `
            <div class="card">
                <h2>📚 累积优化经验</h2>
                <div class="memory-experiences">
                    <div class="prompt-box" style="white-space: pre-wrap; max-height: 500px; overflow-y: auto;">${escapeHtml(data.memory_experiences)}</div>
                </div>
            </div>
            ` : ''}

            <!-- 优化历史（带下拉筛选） -->
            <div class="card">
                <h2>优化历史详情</h2>
                <div class="history-filter">
                    <label for="iterationSelect">选择轮次查看详情：</label>
                    <select id="iterationSelect" onchange="showIterationDetail(this.value)">
                        <option value="">-- 选择轮次 --</option>
                        ${data.history ? data.history.map((item, idx) => {
                            const correctCount = item.total_count - item.error_count;
                            const iterAccuracy = item.total_count > 0 ? ((correctCount / item.total_count) * 100).toFixed(2) : '0.00';
                            return `<option value="${idx}">迭代 ${item.iteration} - 准确率: ${iterAccuracy}%</option>`;
                        }).join('') : ''}
                    </select>
                </div>
                <div id="iterationDetail" class="iteration-detail"></div>
            </div>
        </div>
    `;
    
    // 绘制折线图
    drawAccuracyChart(chartData);
}

function prepareChartData(history) {
    const labels = [];
    const accuracies = [];
    
    history.forEach((item) => {
        const correctCount = item.total_count - item.error_count;
        const iterAccuracy = item.total_count > 0 ? ((correctCount / item.total_count) * 100) : 0;
        labels.push(`迭代 ${item.iteration}`);
        accuracies.push(parseFloat(iterAccuracy.toFixed(2)));
    });
    
    return { labels, accuracies };
}

function drawAccuracyChart(chartData) {
    const ctx = document.getElementById('accuracyChart');
    if (!ctx) return;
    
    // 销毁旧图表
    if (accuracyChart) {
        accuracyChart.destroy();
    }
    
    accuracyChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: chartData.labels,
            datasets: [{
                label: '准确率 (%)',
                data: chartData.accuracies,
                borderColor: '#667eea',
                backgroundColor: 'rgba(102, 126, 234, 0.1)',
                borderWidth: 3,
                fill: true,
                tension: 0.4,
                pointRadius: 6,
                pointBackgroundColor: '#667eea',
                pointBorderColor: '#fff',
                pointBorderWidth: 2,
                pointHoverRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: function(context) {
                            return `准确率: ${context.parsed.y.toFixed(2)}%`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    },
                    title: {
                        display: true,
                        text: '准确率 (%)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: '迭代轮次'
                    }
                }
            }
        }
    });
}

function showIterationDetail(index) {
    const detailDiv = document.getElementById('iterationDetail');
    if (!detailDiv || index === '' || !currentHistoryData || !currentHistoryData[index]) {
        detailDiv.innerHTML = '';
        return;
    }
    
    const item = currentHistoryData[index];
    const correctCount = item.total_count - item.error_count;
    const iterAccuracy = item.total_count > 0 ? ((correctCount / item.total_count) * 100).toFixed(2) : '0.00';
    
    // 显示每轮的预测结果
    let resultsHtml = '';
    if (item.results && item.results.length > 0) {
        resultsHtml = '<div class="iteration-results">';
        resultsHtml += '<h4 style="margin-bottom: 15px; color: #667eea;">本轮预测结果:</h4>';
        resultsHtml += '<div class="results-container">';
        
        item.results.forEach((result, idx) => {
            const isCorrect = result.is_correct;
            const statusIcon = isCorrect ? '✓' : '✗';
            const statusClass = isCorrect ? 'correct' : 'incorrect';
            const inputText = result.input || '';
            const prediction = result.prediction || '';
            const groundTruth = result.ground_truth || '';
            
            resultsHtml += `
                <div class="result-item ${statusClass}">
                    <div class="result-header">
                        <span class="result-icon">${statusIcon}</span>
                        <strong>样本 ${idx + 1}</strong>
                    </div>
                    <div class="result-content">
                        <p><strong>输入:</strong> ${escapeHtml(inputText)}</p>
                        <p><strong>预测:</strong> <span class="prediction ${statusClass}">${escapeHtml(prediction)}</span></p>
                        <p><strong>正确答案:</strong> ${escapeHtml(groundTruth)}</p>
                    </div>
                </div>
            `;
        });
        
        resultsHtml += '</div></div>';
    }
    
    detailDiv.innerHTML = `
        <div class="iteration-detail-content">
            <div class="iteration-header">
                <h3>迭代 ${item.iteration} - 准确率: ${iterAccuracy}% (${correctCount}/${item.total_count})</h3>
            </div>
            
            <div class="iteration-section">
                <h4>当前 Prompt</h4>
                <div class="prompt-box">${escapeHtml(item.prompt)}</div>
            </div>
            
            ${resultsHtml}
            
            <div class="iteration-section">
                <h4>改进建议</h4>
                <div class="prompt-box">${escapeHtml(item.suggestions)}</div>
            </div>
        </div>
    `;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

