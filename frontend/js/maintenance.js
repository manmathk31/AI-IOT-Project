/* ===========================
   MAINTENANCE TAB
   =========================== */

let allTasks = [];

async function initMaintenance() {
  setupMaintenanceForm();
  await fetchTasks();
}

function setupMaintenanceForm() {
  const form = document.getElementById('maintenance-form');
  if (!form) return;

  form.removeEventListener('submit', handleFormSubmit);
  form.addEventListener('submit', handleFormSubmit);
}

async function handleFormSubmit(e) {
  e.preventDefault();

  const engineer = document.getElementById('input-engineer').value.trim();
  const machine = document.getElementById('input-machine').value.trim();
  const notes = document.getElementById('input-notes').value.trim();

  if (!engineer || !machine || !notes) return;

  await submitTask(engineer, machine, notes);
}

async function fetchTasks() {
  try {
    const res = await fetch('http://localhost:8000/api/maintenance');
    if (!res.ok) throw new Error('Maintenance fetch failed');
    const data = await res.json();

    allTasks = data.tasks || [];
    renderKanban();

  } catch (err) {
    console.error('Fetch tasks error:', err);
  }
}

function renderKanban() {
  const pending = allTasks.filter(t => t.status === 'Pending');
  const inProgress = allTasks.filter(t => t.status === 'In Progress');
  const completed = allTasks.filter(t => t.status === 'Completed');

  const countPending = document.getElementById('count-pending');
  const countProgress = document.getElementById('count-progress');
  const countCompleted = document.getElementById('count-completed');

  if (countPending) countPending.textContent = pending.length;
  if (countProgress) countProgress.textContent = inProgress.length;
  if (countCompleted) countCompleted.textContent = completed.length;

  const emptyState = document.getElementById('maintenance-empty');
  const board = document.getElementById('kanban-board');

  if (allTasks.length === 0) {
    if (emptyState) emptyState.classList.remove('hidden');
    if (board) board.style.display = 'none';
    return;
  }

  if (emptyState) emptyState.classList.add('hidden');
  if (board) board.style.display = 'grid';

  renderColumn('cards-pending', pending, 'pending');
  renderColumn('cards-progress', inProgress, 'progress');
  renderColumn('cards-completed', completed, 'completed');
}

function renderColumn(containerId, tasks, type) {
  const container = document.getElementById(containerId);
  if (!container) return;

  if (tasks.length === 0) {
    container.innerHTML = '<div class="empty-state" style="padding:24px 16px;"><p style="font-size:12px;color:#9CA3AF;">No tasks</p></div>';
    return;
  }

  let html = '';
  tasks.forEach(task => {
    let timeFormatted = '';
    try {
      const d = new Date(task.timestamp);
      timeFormatted = d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }) + ' ' +
        d.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit' });
    } catch (e) {
      timeFormatted = task.timestamp || '';
    }

    const notes = task.notes || '';
    const truncNotes = notes.length > 80 ? notes.substring(0, 80) + '...' : notes;

    const cardClass = type === 'pending' ? 'kanban-card-pending'
      : type === 'progress' ? 'kanban-card-progress'
      : 'kanban-card-completed';

    let actionHtml = '';
    if (type === 'pending') {
      actionHtml = '<button class="btn-start" onclick="updateTaskStatus(' + task.id + ', \'In Progress\')">Start Work</button>';
    } else if (type === 'progress') {
      actionHtml = '<button class="btn-complete" onclick="updateTaskStatus(' + task.id + ', \'Completed\')">Mark Complete</button>';
    } else {
      actionHtml = '<span class="completed-check"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#16A34A" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg> Done</span>';
    }

    html += '<div class="kanban-card ' + cardClass + '">';
    html += '<div class="kanban-card-engineer">' + escapeHtml(task.engineer || '') + '</div>';
    html += '<div class="kanban-card-machine">' + escapeHtml(task.machine || '') + '</div>';
    html += '<div class="kanban-card-notes">' + escapeHtml(truncNotes) + '</div>';
    html += '<div class="kanban-card-footer">';
    html += '<span class="kanban-card-time">' + timeFormatted + '</span>';
    html += actionHtml;
    html += '</div>';
    html += '</div>';
  });

  container.innerHTML = html;
}

async function submitTask(engineer, machine, notes) {
  try {
    const res = await fetch('http://localhost:8000/api/maintenance', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ engineer, machine, notes })
    });
    if (!res.ok) throw new Error('Submit task failed');

    document.getElementById('input-engineer').value = '';
    document.getElementById('input-machine').value = '';
    document.getElementById('input-notes').value = '';

    await fetchTasks();
  } catch (err) {
    console.error('Submit task error:', err);
  }
}

async function updateTaskStatus(id, status) {
  try {
    const res = await fetch('http://localhost:8000/api/maintenance/' + id, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ status })
    });
    if (!res.ok) throw new Error('Update status failed');
    await fetchTasks();
  } catch (err) {
    console.error('Update task status error:', err);
  }
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}
