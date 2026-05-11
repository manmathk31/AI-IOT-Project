/* ===========================
   AI MODEL CONTROL PANEL
   =========================== */

let featureChart = null;
let insightsLoaded = false;
let gaugeInterval = null;
let statusInterval = null;

// Keep track of state locally
let isCollecting = false;

async function initInsights() {
  try {
    // Initial fetch of metrics + status
    await fetchModelStatus();
    
    const metricsRes = await fetch('/api/model-metrics');
    if (metricsRes.ok) {
      const data = await metricsRes.json();
      updateMetricCards(data);
      renderFeatureImportances(data.feature_importances);
    }

    // Live gauge and prediction polling (every 2s)
    if (gaugeInterval) clearInterval(gaugeInterval);
    gaugeInterval = setInterval(async () => {
      await updateAnomalyGauge();
      await fetchInsightPrediction();
    }, 2000);
    
    // Status polling for collection progress (every 1s)
    if (statusInterval) clearInterval(statusInterval);
    statusInterval = setInterval(fetchModelStatus, 1000);

    if (typeof activeIntervals !== 'undefined') {
      activeIntervals.push(gaugeInterval);
      activeIntervals.push(statusInterval);
    }

  } catch (err) {
    console.error('AI Insights init error:', err);
  }
}

/* ── MODEL STATUS & CONFIG POLLING ── */
async function fetchModelStatus() {
  try {
    const res = await fetch('/api/model/status');
    if (!res.ok) return;
    const data = await res.json();

    // 1. Update Model Banner
    const banner = document.getElementById('model-status-banner');
    const title = document.getElementById('model-status-title');
    const desc = document.getElementById('model-status-desc');
    const btnReload = document.getElementById('btn-reload-model');
    const btnDelete = document.getElementById('btn-delete-model');

    if (banner) {
      banner.style.display = 'flex';
      if (data.model.model_loaded) {
        banner.className = 'model-status-banner status-loaded';
        title.textContent = 'Model Loaded & Active';
        
        let dateStr = 'Unknown';
        if (data.model.trained_at) {
          dateStr = new Date(data.model.trained_at).toLocaleString();
        }
        desc.textContent = `Trained on ${dateStr} with ${data.model.training_samples} normal samples.`;
        
        if (btnReload) btnReload.style.display = 'inline-block';
        if (btnDelete) btnDelete.style.display = 'inline-block';
      } else {
        banner.className = 'model-status-banner status-missing';
        title.textContent = 'No Model Loaded';
        desc.textContent = 'System is running in simulation mode. Collect data and train a model to activate real-time detection.';
        
        if (btnReload) btnReload.style.display = 'none';
        if (btnDelete) btnDelete.style.display = 'none';
      }
    }

    // 2. Update Sensor Config
    if (data.sensor_config) {
      const tFlame = document.getElementById('toggle-flame');
      const tVib = document.getElementById('toggle-vibration');
      const textFlame = document.getElementById('flame-status-text');
      const textVib = document.getElementById('vibration-status-text');
      
      // Only update checkboxes if they aren't currently focused to prevent jumping
      if (tFlame && document.activeElement !== tFlame) {
        tFlame.checked = data.sensor_config.flame_enabled;
        if (textFlame) textFlame.textContent = data.sensor_config.flame_enabled ? 'Enabled' : 'Disabled';
      }
      if (tVib && document.activeElement !== tVib) {
        tVib.checked = data.sensor_config.vibration_enabled;
        if (textVib) textVib.textContent = data.sensor_config.vibration_enabled ? 'Enabled' : 'Disabled';
      }
    }

    // 3. Update Collection Stats
    const col = data.collection;
    isCollecting = col.status === 'collecting';
    
    document.getElementById('col-status').textContent = isCollecting ? 'Collecting...' : 'Idle';
    document.getElementById('col-status').style.color = isCollecting ? '#2563EB' : '#6B7280';
    
    document.getElementById('col-readings').textContent = `${col.readings} / 600`;
    document.getElementById('col-duration').textContent = `${col.duration_seconds}s`;
    
    const progressPct = Math.min(100, (col.readings / 600) * 100);
    document.getElementById('col-progress').style.width = `${progressPct}%`;
    document.getElementById('col-progress').style.background = col.minimum_met ? '#16A34A' : '#2563EB';

    document.getElementById('btn-start-col').disabled = isCollecting;
    document.getElementById('btn-stop-col').disabled = !isCollecting;
    document.getElementById('btn-train').disabled = isCollecting; // Prevent training while collecting

    // 4. Update Training Stats
    const trainData = data.training_data;
    document.getElementById('train-files').textContent = `${trainData.total_files} file(s)`;
    document.getElementById('train-rows').textContent = trainData.total_rows.toLocaleString();

  } catch (err) {
    console.error('Status fetch error:', err);
  }
}

async function updateSensorConfig() {
  const flameEnabled = document.getElementById('toggle-flame').checked;
  const vibEnabled = document.getElementById('toggle-vibration').checked;
  
  document.getElementById('flame-status-text').textContent = flameEnabled ? 'Enabled' : 'Disabled';
  document.getElementById('vibration-status-text').textContent = vibEnabled ? 'Enabled' : 'Disabled';

  try {
    const res = await fetch('/api/sensor-config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        flame_enabled: flameEnabled,
        vibration_enabled: vibEnabled
      })
    });
    if (!res.ok) throw new Error('Failed to update config');
  } catch (err) {
    alert('Error updating sensor config: ' + err.message);
  }
}

/* ── DATA COLLECTION ACTIONS ── */
async function startCollection() {
  try {
    const res = await fetch('/api/model/collect-start', { method: 'POST' });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    fetchModelStatus();
  } catch (err) {
    alert('Error: ' + err.message);
  }
}

async function stopCollection() {
  try {
    const res = await fetch('/api/model/collect-stop', { method: 'POST' });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    alert(data.message);
    fetchModelStatus();
  } catch (err) {
    alert('Error: ' + err.message);
  }
}

/* ── MODEL ACTIONS ── */
async function trainModel() {
  const btn = document.getElementById('btn-train');
  btn.textContent = 'Training...';
  btn.disabled = true;

  try {
    const res = await fetch('/api/model/train', { method: 'POST' });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    
    alert(data.message || 'Model trained successfully!');
    
    // Refresh metrics chart
    updateMetricCards(data.metrics);
    renderFeatureImportances(data.metrics.feature_importances);
    fetchModelStatus();
    
  } catch (err) {
    alert('Training Failed: ' + err.message);
  } finally {
    btn.textContent = 'Train Model';
    btn.disabled = false;
  }
}

async function deleteTrainingData() {
  if (!confirm("Are you sure you want to delete all collected training data CSVs?")) return;
  
  try {
    const res = await fetch('/api/model/training-data', { method: 'DELETE' });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    fetchModelStatus();
  } catch (err) {
    alert('Error: ' + err.message);
  }
}

async function reloadModel() {
  try {
    const res = await fetch('/api/model/reload', { method: 'POST' });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    alert('Model reloaded from disk successfully.');
  } catch (err) {
    alert('Error: ' + err.message);
  }
}

async function deleteModel() {
  if (!confirm("WARNING: This will delete the active model and fallback to simulation mode. Continue?")) return;
  
  try {
    const res = await fetch('/api/model/delete', { method: 'POST' });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    fetchModelStatus();
  } catch (err) {
    alert('Error: ' + err.message);
  }
}


/* ── UI RENDERERS ── */
function updateMetricCards(data) {
  if (!data) return;
  const ndrEl = document.getElementById('metric-normal-detection-rate');
  const tsEl = document.getElementById('metric-training-samples');
  const wtEl = document.getElementById('metric-warning-threshold');
  const ftEl = document.getElementById('metric-fault-threshold');

  if (ndrEl) ndrEl.textContent = data.normal_detection_rate
      ? (data.normal_detection_rate * 100).toFixed(1) + '%' : '—';
  if (tsEl) tsEl.textContent = data.training_samples != null
      ? data.training_samples.toLocaleString() : '—';
  if (wtEl) wtEl.textContent = data.threshold_warning != null
      ? data.threshold_warning.toFixed(1) : '-0.2';
  if (ftEl) ftEl.textContent = data.threshold_fault != null
      ? data.threshold_fault.toFixed(1) : '-0.5';
}

async function updateAnomalyGauge() {
  try {
    const res = await fetch('/api/prediction');
    if (!res.ok) return;
    const data = await res.json();

    const score = data.anomaly_score;
    const gaugeBar = document.getElementById('gauge-indicator');
    const gaugeValue = document.getElementById('gauge-score-value');
    const gaugeLabel = document.getElementById('gauge-score-label');

    if (!gaugeBar) return;

    // Gauge range: -1.5 (left, most anomalous) to +1.0 (right, most normal)
    const minScore = -1.5;
    const maxScore = 1.0;
    const range = maxScore - minScore; // 2.5

    if (score !== null && score !== undefined) {
      const clampedScore = Math.max(minScore, Math.min(maxScore, score));
      const pct = ((clampedScore - minScore) / range) * 100;

      gaugeBar.style.left = pct + '%';
      gaugeBar.style.opacity = '1';

      if (gaugeValue) gaugeValue.textContent = score.toFixed(3);
      if (gaugeLabel) {
        if (score > -0.2) {
          gaugeLabel.textContent = 'Normal';
          gaugeLabel.className = 'gauge-label gauge-label-normal';
        } else if (score > -0.5) {
          gaugeLabel.textContent = 'Warning';
          gaugeLabel.className = 'gauge-label gauge-label-warning';
        } else {
          gaugeLabel.textContent = 'Fault';
          gaugeLabel.className = 'gauge-label gauge-label-fault';
        }
      }
    } else {
      gaugeBar.style.opacity = '0.3';
      if (gaugeValue) gaugeValue.textContent = '—';
      if (gaugeLabel) {
        gaugeLabel.textContent = data.prediction || 'N/A';
        gaugeLabel.className = 'gauge-label';
      }
    }
  } catch (err) {
    console.error('Gauge update error:', err);
  }
}

function renderFeatureImportances(importances) {
  if (!importances) return;

  const entries = Object.entries(importances).sort((a, b) => b[1] - a[1]);
  const labels = entries.map(e => e[0].replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()));
  const values = entries.map(e => e[1]);

  const ctx = document.getElementById('feature-chart');
  if (!ctx) return;

  if (featureChart) {
    featureChart.destroy();
    featureChart = null;
  }

  featureChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Importance',
        data: values,
        backgroundColor: 'rgba(37, 99, 235, 0.8)',
        borderColor: '#2563EB',
        borderWidth: 1,
        borderRadius: 4,
        barThickness: 22
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
      },
      scales: {
        x: {
          grid: { color: '#F3F4F6', drawBorder: false },
          ticks: { font: { family: "'Space Mono', monospace", size: 10 }, color: '#9CA3AF' },
          border: { display: false },
          max: Math.max(0.25, ...values.map(v => v * 1.2))
        },
        y: {
          grid: { display: false },
          ticks: { font: { family: "'DM Sans', sans-serif", size: 12 }, color: '#374151' },
          border: { display: false }
        }
      }
    }
  });
}

async function fetchInsightPrediction() {
  try {
    const res = await fetch('/api/prediction');
    if (!res.ok) return;
    const data = await res.json();

    const badge = document.getElementById('insight-prediction-badge');
    const text = document.getElementById('insight-prediction-text');
    const bar = document.getElementById('insight-confidence-bar');
    const label = document.getElementById('insight-confidence-label');

    if (badge) {
      badge.className = 'status-badge';
      const pred = (data.prediction || '').toUpperCase();
      if (pred.includes('CRITICAL') || pred.includes('FLAME')) {
        badge.classList.add('badge-critical');
      } else if (pred.includes('FAULT')) {
        badge.classList.add('badge-fault');
      } else if (pred.includes('WARNING') || pred.includes('VIBRATION')) {
        badge.classList.add('badge-warning');
      } else {
        badge.classList.add('badge-normal');
      }
    }

    if (text) text.textContent = (data.prediction || 'Loading...').replace(/_/g, ' ');

    const conf = data.confidence != null ? data.confidence * 100 : 0;
    if (bar) bar.style.width = conf.toFixed(1) + '%';
    if (label) label.textContent = conf.toFixed(1) + '%';

  } catch (err) {
    console.error('Insight prediction fetch error:', err);
  }
}

