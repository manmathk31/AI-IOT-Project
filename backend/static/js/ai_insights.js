/* ===========================
   AI INSIGHTS TAB
   =========================== */

let featureChart = null;
let insightsLoaded = false;

async function initInsights() {
  try {
    const metricsRes = await fetch('/api/model-metrics');
    if (!metricsRes.ok) throw new Error('Model metrics fetch failed');
    const data = await metricsRes.json();

    const accEl = document.getElementById('metric-accuracy');
    const f1El = document.getElementById('metric-f1');
    const precEl = document.getElementById('metric-precision');
    const recEl = document.getElementById('metric-recall');

    if (accEl) accEl.textContent = (data.accuracy * 100).toFixed(1) + '%';
    if (f1El) f1El.textContent = (data.f1 * 100).toFixed(1) + '%';
    if (precEl) precEl.textContent = (data.precision * 100).toFixed(1) + '%';
    if (recEl) recEl.textContent = (data.recall * 100).toFixed(1) + '%';

    renderConfusionMatrix(data.confusion_matrix);
    renderFeatureImportances(data.feature_importances);

    await fetchInsightPrediction();

  } catch (err) {
    console.error('AI Insights init error:', err);
  }
}

function renderConfusionMatrix(matrix) {
  const container = document.getElementById('confusion-matrix-container');
  if (!container || !matrix) return;

  const labels = ['Normal', 'Warning', 'Fault', 'C.Flame'];
  let maxVal = 0;
  matrix.forEach(row => row.forEach(val => { if (val > maxVal) maxVal = val; }));

  let html = '<table class="confusion-table">';

  html += '<thead><tr><th></th>';
  labels.forEach(l => { html += '<th>' + l + '</th>'; });
  html += '</tr></thead>';

  html += '<tbody>';
  matrix.forEach((row, i) => {
    html += '<tr>';
    html += '<th style="background:#F9FAFB;font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:0.05em;color:#6B7280;padding:12px 8px;border:1px solid #E5E7EB;text-align:left;">Actual ' + labels[i] + '</th>';
    row.forEach(val => {
      const intensity = maxVal > 0 ? val / maxVal : 0;
      const bgAlpha = intensity * 0.7;
      const bg = 'rgba(37, 99, 235, ' + bgAlpha.toFixed(2) + ')';
      const textColor = intensity > 0.5 ? '#FFFFFF' : '#111827';
      html += '<td style="background:' + bg + ';color:' + textColor + ';">' + val + '</td>';
    });
    html += '</tr>';
  });
  html += '</tbody></table>';

  container.innerHTML = html;
}

function renderFeatureImportances(importances) {
  if (!importances) return;

  const entries = Object.entries(importances).sort((a, b) => b[1] - a[1]);
  const labels = entries.map(e => {
    return e[0].replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
  });
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
        tooltip: {
          backgroundColor: '#111827',
          titleFont: { family: "'DM Sans', sans-serif", size: 12 },
          bodyFont: { family: "'Space Mono', monospace", size: 12 },
          padding: 10,
          cornerRadius: 8,
          callbacks: {
            label: function(ctx) {
              return 'Importance: ' + ctx.parsed.x.toFixed(2);
            }
          }
        }
      },
      scales: {
        x: {
          grid: { color: '#F3F4F6', drawBorder: false },
          ticks: {
            font: { family: "'Space Mono', monospace", size: 10 },
            color: '#9CA3AF'
          },
          border: { display: false },
          max: 0.25
        },
        y: {
          grid: { display: false },
          ticks: {
            font: { family: "'DM Sans', sans-serif", size: 12 },
            color: '#374151'
          },
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
      } else if (pred.includes('WARNING')) {
        badge.classList.add('badge-warning');
      } else {
        badge.classList.add('badge-normal');
      }
    }

    if (text) {
      text.textContent = (data.prediction || 'Loading...').replace(/_/g, ' ');
    }

    const conf = data.confidence != null ? data.confidence * 100 : 0;
    if (bar) bar.style.width = conf.toFixed(1) + '%';
    if (label) label.textContent = conf.toFixed(1) + '%';

  } catch (err) {
    console.error('Insight prediction fetch error:', err);
  }
}
