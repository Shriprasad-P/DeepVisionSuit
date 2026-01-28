/* ========================================
   DeepVision Dashboard - JavaScript
   ======================================== */

document.addEventListener('DOMContentLoaded', () => {
    initParticles();
    initNetworkAnimation();
    initNavigation();
    initVisualizationTabs();
    initImageUpload();
    checkAPIStatus();
    initCharts();
    initScrollAnimations();
});

/* ==========================================
   1. Particle Background
   ========================================== */
function initParticles() {
    const container = document.getElementById('particles');
    const particleCount = 50;

    for (let i = 0; i < particleCount; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.cssText = `
            position: absolute;
            width: ${Math.random() * 4 + 1}px;
            height: ${Math.random() * 4 + 1}px;
            background: rgba(99, 102, 241, ${Math.random() * 0.5 + 0.2});
            border-radius: 50%;
            left: ${Math.random() * 100}%;
            top: ${Math.random() * 100}%;
            animation: floatParticle ${Math.random() * 20 + 10}s linear infinite;
            animation-delay: ${Math.random() * 10}s;
        `;
        container.appendChild(particle);
    }

    // Add particle animation CSS
    const style = document.createElement('style');
    style.textContent = `
        @keyframes floatParticle {
            0% { transform: translate(0, 0) rotate(0deg); opacity: 0; }
            10% { opacity: 1; }
            90% { opacity: 1; }
            100% { transform: translate(${Math.random() > 0.5 ? '' : '-'}100px, -100vh) rotate(360deg); opacity: 0; }
        }
    `;
    document.head.appendChild(style);
}

/* ==========================================
   2. Neural Network Animation
   ========================================== */
function initNetworkAnimation() {
    const canvas = document.getElementById('networkCanvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    let animationId;

    // Set canvas size
    function resize() {
        const rect = canvas.parentElement.getBoundingClientRect();
        canvas.width = rect.width;
        canvas.height = rect.height;
    }
    resize();
    window.addEventListener('resize', resize);

    // Network nodes
    const layers = [3, 8, 12, 16, 12, 8, 4];
    let nodes = [];
    let connections = [];
    let pulses = [];

    function initNetwork() {
        nodes = [];
        connections = [];

        const layerSpacing = canvas.width / (layers.length + 1);

        layers.forEach((nodeCount, layerIdx) => {
            const x = layerSpacing * (layerIdx + 1);
            const nodeSpacing = canvas.height / (nodeCount + 1);

            for (let i = 0; i < nodeCount; i++) {
                const y = nodeSpacing * (i + 1);
                nodes.push({
                    x, y,
                    layer: layerIdx,
                    radius: 4 + Math.random() * 2,
                    pulse: Math.random() * Math.PI * 2,
                    active: false
                });
            }
        });

        // Create connections
        let nodeIdx = 0;
        for (let l = 0; l < layers.length - 1; l++) {
            const currentLayerStart = nodeIdx;
            const currentLayerEnd = nodeIdx + layers[l];
            const nextLayerStart = currentLayerEnd;
            const nextLayerEnd = nextLayerStart + layers[l + 1];

            for (let i = currentLayerStart; i < currentLayerEnd; i++) {
                // Connect to some nodes in next layer
                const connectionCount = Math.min(3, layers[l + 1]);
                for (let j = 0; j < connectionCount; j++) {
                    const targetIdx = nextLayerStart + Math.floor(Math.random() * layers[l + 1]);
                    connections.push({
                        from: i,
                        to: targetIdx,
                        weight: Math.random()
                    });
                }
            }
            nodeIdx += layers[l];
        }
    }

    function createPulse() {
        if (connections.length === 0) return;
        const conn = connections[Math.floor(Math.random() * connections.length)];
        pulses.push({
            connection: conn,
            progress: 0,
            speed: 0.02 + Math.random() * 0.02
        });
    }

    function draw() {
        ctx.fillStyle = 'rgba(26, 26, 46, 0.1)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Draw connections
        connections.forEach(conn => {
            const from = nodes[conn.from];
            const to = nodes[conn.to];

            ctx.beginPath();
            ctx.moveTo(from.x, from.y);
            ctx.lineTo(to.x, to.y);
            ctx.strokeStyle = `rgba(99, 102, 241, ${0.1 + conn.weight * 0.1})`;
            ctx.lineWidth = 1;
            ctx.stroke();
        });

        // Draw pulses
        pulses.forEach((pulse, idx) => {
            const from = nodes[pulse.connection.from];
            const to = nodes[pulse.connection.to];

            const x = from.x + (to.x - from.x) * pulse.progress;
            const y = from.y + (to.y - from.y) * pulse.progress;

            ctx.beginPath();
            ctx.arc(x, y, 3, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(236, 72, 153, ${1 - pulse.progress})`;
            ctx.fill();

            // Glow effect
            const gradient = ctx.createRadialGradient(x, y, 0, x, y, 15);
            gradient.addColorStop(0, 'rgba(236, 72, 153, 0.5)');
            gradient.addColorStop(1, 'rgba(236, 72, 153, 0)');
            ctx.fillStyle = gradient;
            ctx.beginPath();
            ctx.arc(x, y, 15, 0, Math.PI * 2);
            ctx.fill();

            pulse.progress += pulse.speed;
            if (pulse.progress >= 1) {
                pulses.splice(idx, 1);
            }
        });

        // Draw nodes
        nodes.forEach((node, idx) => {
            node.pulse += 0.02;
            const pulseScale = 1 + Math.sin(node.pulse) * 0.2;

            // Glow
            const gradient = ctx.createRadialGradient(
                node.x, node.y, 0,
                node.x, node.y, node.radius * 3 * pulseScale
            );
            gradient.addColorStop(0, 'rgba(99, 102, 241, 0.5)');
            gradient.addColorStop(1, 'rgba(99, 102, 241, 0)');
            ctx.fillStyle = gradient;
            ctx.beginPath();
            ctx.arc(node.x, node.y, node.radius * 3 * pulseScale, 0, Math.PI * 2);
            ctx.fill();

            // Node
            ctx.beginPath();
            ctx.arc(node.x, node.y, node.radius * pulseScale, 0, Math.PI * 2);

            const nodeGradient = ctx.createRadialGradient(
                node.x - node.radius * 0.3, node.y - node.radius * 0.3, 0,
                node.x, node.y, node.radius * 2
            );
            nodeGradient.addColorStop(0, '#8b8ff5');
            nodeGradient.addColorStop(1, '#6366f1');
            ctx.fillStyle = nodeGradient;
            ctx.fill();
        });

        // Create new pulses
        if (Math.random() < 0.1) {
            createPulse();
        }

        animationId = requestAnimationFrame(draw);
    }

    initNetwork();
    draw();
}

/* ==========================================
   3. Navigation
   ========================================== */
function initNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    const sections = document.querySelectorAll('section[id]');

    // Smooth scroll
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = link.getAttribute('href');
            const target = document.querySelector(targetId);
            if (target) {
                target.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });

    // Active state on scroll
    window.addEventListener('scroll', () => {
        let current = '';

        sections.forEach(section => {
            const sectionTop = section.offsetTop - 100;
            const sectionHeight = section.clientHeight;

            if (window.scrollY >= sectionTop && window.scrollY < sectionTop + sectionHeight) {
                current = section.getAttribute('id');
            }
        });

        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === `#${current}`) {
                link.classList.add('active');
            }
        });
    });
}

/* ==========================================
   4. Visualization Tabs
   ========================================== */
function initVisualizationTabs() {
    const tabs = document.querySelectorAll('.viz-tab');
    const panels = document.querySelectorAll('.viz-panel');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const targetPanel = tab.dataset.tab;

            // Update tabs
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');

            // Update panels
            panels.forEach(p => p.classList.remove('active'));
            document.getElementById(targetPanel).classList.add('active');
        });
    });
}

/* ==========================================
   5. Image Upload & Inference
   ========================================== */
function initImageUpload() {
    const uploadArea = document.getElementById('uploadArea');
    const imageInput = document.getElementById('imageInput');
    const previewArea = document.getElementById('previewArea');
    const previewImage = document.getElementById('previewImage');
    const classifyBtn = document.getElementById('classifyBtn');
    const clearBtn = document.getElementById('clearBtn');
    const resultsArea = document.getElementById('resultsArea');

    let currentFile = null;

    // Click to upload
    uploadArea.addEventListener('click', () => {
        imageInput.click();
    });

    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');

        const files = e.dataTransfer.files;
        if (files.length > 0 && files[0].type.startsWith('image/')) {
            handleFile(files[0]);
        }
    });

    // File input change
    imageInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        currentFile = file;

        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            uploadArea.style.display = 'none';
            previewArea.classList.add('active');
            resultsArea.classList.remove('active');
        };
        reader.readAsDataURL(file);
    }

    // Clear button
    clearBtn.addEventListener('click', () => {
        currentFile = null;
        previewImage.src = '';
        uploadArea.style.display = 'block';
        previewArea.classList.remove('active');
        resultsArea.classList.remove('active');
        imageInput.value = '';
    });

    // Classify button
    classifyBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        classifyBtn.disabled = true;
        classifyBtn.innerHTML = '<span class="btn-icon">⏳</span> Classifying...';

        try {
            const formData = new FormData();
            formData.append('file', currentFile);

            const response = await fetch('http://localhost:8000/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Inference failed');
            }

            const data = await response.json();
            displayResults(data);

        } catch (error) {
            console.error('Error:', error);
            alert('Failed to classify image. Make sure the API server is running on localhost:8000');
        } finally {
            classifyBtn.disabled = false;
            classifyBtn.innerHTML = '<span class="btn-icon">🔍</span> Classify Image';
        }
    });
}

function displayResults(data) {
    const resultsArea = document.getElementById('resultsArea');
    const inferenceTime = document.getElementById('inferenceTime');
    const topPrediction = document.getElementById('topPrediction');
    const confidenceFill = document.getElementById('confidenceFill');
    const confidenceValue = document.getElementById('confidenceValue');
    const top5List = document.querySelector('.prediction-list');

    resultsArea.classList.add('active');

    // Inference time
    inferenceTime.querySelector('.time-value').textContent = `${data.inference_time_ms} ms`;

    // Top prediction
    topPrediction.querySelector('.prediction-class').textContent =
        `Class ID: ${data.prediction.class_id}`;

    const confidence = (data.prediction.confidence * 100).toFixed(1);
    confidenceFill.style.width = `${confidence}%`;
    confidenceValue.textContent = `${confidence}%`;

    // Top 5 predictions
    top5List.innerHTML = data.top5.map((pred, idx) => `
        <div class="prediction-item">
            <span class="class-id">${idx + 1}. Class ${pred.class_id}</span>
            <span class="confidence">${(pred.confidence * 100).toFixed(2)}%</span>
        </div>
    `).join('');
}

/* ==========================================
   6. API Status Check
   ========================================== */
async function checkAPIStatus() {
    const statusIndicator = document.getElementById('apiStatus');
    const statusDot = statusIndicator.querySelector('.status-dot');
    const statusText = statusIndicator.querySelector('.status-text');

    try {
        const response = await fetch('http://localhost:8000/health', {
            method: 'GET',
            mode: 'cors'
        });

        if (response.ok) {
            statusIndicator.classList.add('online');
            statusText.textContent = 'API Online';
        } else {
            throw new Error('API not healthy');
        }
    } catch (error) {
        statusIndicator.classList.remove('online');
        statusText.textContent = 'API Offline';
    }

    // Check status every 30 seconds
    setTimeout(checkAPIStatus, 30000);
}

/* ==========================================
   7. Charts (Simulated Training Metrics)
   ========================================== */
function initCharts() {
    drawLossChart();
    drawAccuracyChart();
    drawLRChart();
}

function drawLossChart() {
    const canvas = document.getElementById('lossCanvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;

    // Simulated loss data
    const trainLoss = [3.5, 2.8, 2.3, 1.9, 1.6, 1.4, 1.2, 1.1, 1.0, 0.9];
    const valLoss = [3.6, 3.0, 2.5, 2.2, 1.9, 1.7, 1.5, 1.4, 1.35, 1.3];

    drawLineChart(ctx, canvas, [
        { data: trainLoss, color: '#6366f1', label: 'Train Loss' },
        { data: valLoss, color: '#ec4899', label: 'Val Loss' }
    ], 'Loss');
}

function drawAccuracyChart() {
    const canvas = document.getElementById('accCanvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;

    // Simulated accuracy data
    const trainAcc = [15, 28, 38, 48, 55, 62, 68, 72, 75, 78];
    const valAcc = [12, 24, 32, 42, 50, 56, 61, 65, 68, 70];

    drawLineChart(ctx, canvas, [
        { data: trainAcc, color: '#22c55e', label: 'Train Acc' },
        { data: valAcc, color: '#06b6d4', label: 'Val Acc' }
    ], 'Accuracy (%)');
}

function drawLRChart() {
    const canvas = document.getElementById('lrCanvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;

    // Cosine annealing schedule
    const lr = [];
    for (let i = 0; i < 10; i++) {
        lr.push(0.001 * (1 + Math.cos(Math.PI * i / 10)) / 2);
    }

    drawLineChart(ctx, canvas, [
        { data: lr, color: '#f59e0b', label: 'Learning Rate' }
    ], 'LR');
}

function drawLineChart(ctx, canvas, datasets, yLabel) {
    const padding = { top: 30, right: 20, bottom: 40, left: 50 };
    const width = canvas.width - padding.left - padding.right;
    const height = canvas.height - padding.top - padding.bottom;

    // Clear
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Find min/max
    let maxVal = -Infinity;
    let minVal = Infinity;
    datasets.forEach(ds => {
        maxVal = Math.max(maxVal, ...ds.data);
        minVal = Math.min(minVal, ...ds.data);
    });

    const range = maxVal - minVal || 1;

    // Grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
        const y = padding.top + (height * i / 4);
        ctx.beginPath();
        ctx.moveTo(padding.left, y);
        ctx.lineTo(padding.left + width, y);
        ctx.stroke();
    }

    // Draw datasets
    datasets.forEach(ds => {
        ctx.strokeStyle = ds.color;
        ctx.lineWidth = 2;
        ctx.beginPath();

        ds.data.forEach((val, i) => {
            const x = padding.left + (width * i / (ds.data.length - 1));
            const y = padding.top + height - ((val - minVal) / range * height);

            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });

        ctx.stroke();

        // Draw points
        ds.data.forEach((val, i) => {
            const x = padding.left + (width * i / (ds.data.length - 1));
            const y = padding.top + height - ((val - minVal) / range * height);

            ctx.beginPath();
            ctx.arc(x, y, 4, 0, Math.PI * 2);
            ctx.fillStyle = ds.color;
            ctx.fill();
        });
    });

    // Axis labels
    ctx.fillStyle = '#6b6b80';
    ctx.font = '12px Inter';
    ctx.textAlign = 'center';
    ctx.fillText('Epochs', padding.left + width / 2, canvas.height - 10);

    ctx.save();
    ctx.translate(15, padding.top + height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(yLabel, 0, 0);
    ctx.restore();

    // Legend
    let legendX = padding.left + 10;
    datasets.forEach(ds => {
        ctx.fillStyle = ds.color;
        ctx.fillRect(legendX, 10, 15, 10);
        ctx.fillStyle = '#a0a0b0';
        ctx.font = '11px Inter';
        ctx.textAlign = 'left';
        ctx.fillText(ds.label, legendX + 20, 18);
        legendX += ctx.measureText(ds.label).width + 40;
    });
}

/* ==========================================
   8. Scroll Animations
   ========================================== */
function initScrollAnimations() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
            }
        });
    }, {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    });

    document.querySelectorAll('.animated-card').forEach(card => {
        observer.observe(card);
    });
}

/* ==========================================
   9. Utility Functions
   ========================================== */

// Format large numbers
function formatNumber(num) {
    if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
}

// Debounce function
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Handle resize
window.addEventListener('resize', debounce(() => {
    initCharts();
}, 250));
