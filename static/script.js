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
        optimizeBtn.disabled = false;
        optimizeBtn.textContent = '开始优化';

        if (data.error) {
            resultArea.innerHTML = `
                <div class="error-message">
                    <strong>错误:</strong> ${data.error}
                </div>
            `;
            return;
        }

        displayResults(data);
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

