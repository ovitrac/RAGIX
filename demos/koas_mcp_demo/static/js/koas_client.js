/**
 * KOAS MCP Demo Client
 *
 * Client-side JavaScript for the KOAS demo application.
 * Handles tool execution, WebSocket communication, and UI updates.
 */

// =============================================================================
// STATE
// =============================================================================

let tools = {};
let selectedTool = null;
let ws = null;
let currentWorkspace = null;
let chatHistory = [];  // Track conversation for history sidebar
let sessionMode = 'save';  // 'save' or 'load'

// =============================================================================
// INITIALIZATION
// =============================================================================

document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    loadTools();
    loadModels();
    connectWebSocket();
    loadMemory();

    // Configure marked.js for markdown rendering
    if (typeof marked !== 'undefined') {
        marked.setOptions({
            breaks: true,  // Convert \n to <br>
            gfm: true      // GitHub Flavored Markdown
        });
        console.log('Marked.js configured successfully');
    } else {
        console.warn('Marked.js not loaded');
    }
});

// =============================================================================
// TAB NAVIGATION
// =============================================================================

function initTabs() {
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => {
            // Update tab buttons
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');

            // Update panels
            const targetId = tab.dataset.tab;
            document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
            document.getElementById(targetId).classList.add('active');
        });
    });
}

// =============================================================================
// TOOLS
// =============================================================================

async function loadTools() {
    try {
        const response = await fetch('/api/tools');
        const data = await response.json();
        tools = data.tools;
        renderTools();
    } catch (error) {
        console.error('Error loading tools:', error);
        addTraceEntry('error', 'Failed to load tools', error.message);
    }
}

function renderTools() {
    const grid = document.getElementById('toolsGrid');
    const filter = document.getElementById('categoryFilter').value;

    grid.innerHTML = '';

    for (const [name, info] of Object.entries(tools)) {
        if (filter !== 'all' && info.category !== filter) continue;

        const card = document.createElement('div');
        card.className = 'tool-card';
        card.onclick = () => showToolForm(name, info);

        card.innerHTML = `
            <h3>${formatToolName(name)}</h3>
            <p>${info.description}</p>
            <div class="meta">
                <span class="badge ${info.category}">${info.category}</span>
                <span class="badge stage">Stage ${info.stage}</span>
            </div>
        `;

        grid.appendChild(card);
    }
}

function formatToolName(name) {
    return name
        .replace('koas_security_', 'Security: ')
        .replace('koas_audit_', 'Audit: ')
        .replace(/_/g, ' ')
        .replace(/\b\w/g, c => c.toUpperCase());
}

function showToolForm(name, info) {
    selectedTool = name;
    const form = document.getElementById('toolForm');
    const title = document.getElementById('toolFormTitle');
    const params = document.getElementById('toolParams');

    title.textContent = formatToolName(name);
    params.innerHTML = '';

    for (const [paramName, paramInfo] of Object.entries(info.parameters)) {
        const group = document.createElement('div');
        group.className = 'form-group';

        let input;
        if (paramInfo.options) {
            input = document.createElement('select');
            input.innerHTML = paramInfo.options.map(opt =>
                `<option value="${opt}" ${opt === paramInfo.default ? 'selected' : ''}>${opt}</option>`
            ).join('');
        } else if (paramInfo.type === 'boolean') {
            input = document.createElement('input');
            input.type = 'checkbox';
            input.checked = paramInfo.default;
        } else if (paramInfo.type === 'integer') {
            input = document.createElement('input');
            input.type = 'number';
            input.value = paramInfo.default || '';
        } else {
            input = document.createElement('input');
            input.type = 'text';
            input.value = paramInfo.default || '';
            if (paramInfo.example) {
                input.placeholder = `e.g., ${paramInfo.example}`;
            }
        }

        input.id = `param_${paramName}`;
        input.name = paramName;

        const label = document.createElement('label');
        label.htmlFor = input.id;
        label.textContent = `${paramName}${paramInfo.required ? ' *' : ''}`;

        group.appendChild(label);
        group.appendChild(input);
        params.appendChild(group);
    }

    form.classList.remove('hidden');
}

function hideToolForm() {
    document.getElementById('toolForm').classList.add('hidden');
    selectedTool = null;
}

async function executeTool() {
    if (!selectedTool) return;

    const params = {};
    const form = document.getElementById('toolParams');
    const dryRun = document.getElementById('dryRunMode').checked;

    form.querySelectorAll('input, select').forEach(input => {
        if (input.type === 'checkbox') {
            params[input.name] = input.checked;
        } else if (input.type === 'number') {
            params[input.name] = parseInt(input.value) || undefined;
        } else if (input.value) {
            params[input.name] = input.value;
        }
    });

    // Use current workspace if not specified
    if (!params.workspace && currentWorkspace) {
        params.workspace = currentWorkspace;
    }

    addTraceEntry('info', `Executing ${selectedTool}`, JSON.stringify(params, null, 2));

    try {
        const response = await fetch('/api/tool', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                tool_name: selectedTool,
                parameters: params,
                dry_run: dryRun
            })
        });

        const result = await response.json();

        // Update workspace
        if (result.workspace) {
            currentWorkspace = result.workspace;
            updateWorkspaceDisplay();
        }

        // Add to trace
        const status = result.error ? 'error' : 'success';
        addTraceEntry(status, `${selectedTool} completed`, JSON.stringify(result, null, 2));

        // Show result
        showResult(selectedTool, result);

    } catch (error) {
        addTraceEntry('error', `${selectedTool} failed`, error.message);
    }

    hideToolForm();
}

// =============================================================================
// SCENARIOS
// =============================================================================

async function runScenario(scenarioId) {
    const dryRun = document.getElementById('dryRunMode').checked;

    addTraceEntry('info', `Starting scenario: ${scenarioId}`, `Dry run: ${dryRun}`);

    // Switch to trace tab
    document.querySelector('[data-tab="trace"]').click();

    try {
        const response = await fetch('/api/scenario', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                scenario_id: scenarioId,
                dry_run: dryRun
            })
        });

        const result = await response.json();

        // Update workspace
        if (result.workspace) {
            currentWorkspace = result.workspace;
            updateWorkspaceDisplay();
        }

        // Log each step
        result.steps.forEach(step => {
            const status = step.result.error ? 'warning' : 'success';
            addTraceEntry(status, `Step ${step.step}: ${step.tool}`, JSON.stringify(step.result, null, 2));
        });

        addTraceEntry('success', `Scenario ${scenarioId} completed`, `Workspace: ${result.workspace}`);

    } catch (error) {
        addTraceEntry('error', `Scenario ${scenarioId} failed`, error.message);
    }
}

// =============================================================================
// CHAT
// =============================================================================

async function loadModels() {
    try {
        const response = await fetch('/api/models');
        const data = await response.json();

        const select = document.getElementById('modelSelect');
        select.innerHTML = '';

        if (data.models.length > 0) {
            data.models.forEach(model => {
                const option = document.createElement('option');
                option.value = model.name;
                option.textContent = `${model.name} (${model.size})`;
                select.appendChild(option);
            });
        } else {
            // Use recommended as fallback
            data.recommended.forEach(model => {
                const option = document.createElement('option');
                option.value = model;
                option.textContent = model;
                select.appendChild(option);
            });
        }
    } catch (error) {
        console.error('Error loading models:', error);
    }
}

async function refreshModels() {
    await loadModels();
}

function handleChatKeypress(event) {
    if (event.key === 'Enter') {
        sendChat();
    }
}

async function sendChat() {
    const input = document.getElementById('chatInput');
    const message = input.value.trim();
    if (!message) return;

    const model = document.getElementById('modelSelect').value;
    const dryRun = document.getElementById('dryRunMode').checked;

    // Add user message
    addChatMessage('user', message);
    input.value = '';

    // Show loading
    const loadingId = 'loading_' + Date.now();
    addChatMessage('assistant', '<span class="loading"></span> Thinking...', loadingId);

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message,
                model,
                dry_run: dryRun,
                workspace: currentWorkspace
            })
        });

        const data = await response.json();

        // Remove loading and add response
        document.getElementById(loadingId)?.remove();
        addChatMessage('assistant', data.response);

    } catch (error) {
        document.getElementById(loadingId)?.remove();
        addChatMessage('assistant', `Error: ${error.message}`);
    }
}

function renderMarkdown(text) {
    // Try marked.js first
    if (typeof marked !== 'undefined') {
        try {
            if (typeof marked.parse === 'function') {
                return marked.parse(text);
            } else if (typeof marked === 'function') {
                return marked(text);  // Older API
            }
        } catch (e) {
            console.error('Marked error:', e);
        }
    }
    // Fallback: convert newlines and basic markdown
    return text
        .replace(/\n/g, '<br>')
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/_(.+?)_/g, '<em>$1</em>')
        .replace(/`(.+?)`/g, '<code>$1</code>')
        .replace(/^### (.+)$/gm, '<h3>$1</h3>');
}

function addChatMessage(role, content, id = null, skipHistory = false) {
    const messages = document.getElementById('chatMessages');
    const div = document.createElement('div');
    div.className = `message ${role}`;
    if (id) div.id = id;

    // Create message content wrapper
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    // Render markdown for assistant messages (not loading indicators)
    if (role === 'assistant' && !content.includes('class="loading"')) {
        contentDiv.innerHTML = renderMarkdown(content);
    } else {
        contentDiv.innerHTML = `<p>${content}</p>`;
    }

    div.appendChild(contentDiv);
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;

    // Add to history (skip loading messages and duplicates)
    if (!skipHistory && !id) {
        const timestamp = new Date().toLocaleTimeString();
        const preview = content.substring(0, 50) + (content.length > 50 ? '...' : '');
        chatHistory.push({ role, content, preview, timestamp });
        updateHistorySidebar();
    }
}

// =============================================================================
// TRACE LOG
// =============================================================================

function addTraceEntry(type, message, details = null) {
    const log = document.getElementById('traceLog');
    const timestamp = new Date().toLocaleTimeString();

    const entry = document.createElement('div');
    entry.className = `trace-entry ${type}`;

    let html = `
        <span class="timestamp">[${timestamp}]</span>
        <span class="message">${message}</span>
    `;

    if (details) {
        const detailsId = 'details_' + Date.now();
        html += `<span class="toggle" onclick="toggleDetails('${detailsId}')">[+]</span>`;
        html += `<div id="${detailsId}" class="details hidden">${escapeHtml(details)}</div>`;
    }

    entry.innerHTML = html;
    log.appendChild(entry);

    if (document.getElementById('autoScroll').checked) {
        log.scrollTop = log.scrollHeight;
    }
}

function toggleDetails(id) {
    const details = document.getElementById(id);
    details.classList.toggle('hidden');

    const toggle = details.previousElementSibling;
    toggle.textContent = details.classList.contains('hidden') ? '[+]' : '[-]';
}

function clearTrace() {
    document.getElementById('traceLog').innerHTML = `
        <div class="trace-entry info">
            <span class="timestamp">[--:--:--]</span>
            <span class="message">Trace cleared</span>
        </div>
    `;
}

// =============================================================================
// RESULT MODAL
// =============================================================================

function showResult(title, result) {
    document.getElementById('resultTitle').textContent = formatToolName(title);

    const summary = document.getElementById('resultSummary');
    summary.className = 'result-summary ' + (result.error ? 'error' : 'success');
    summary.innerHTML = `<strong>${result.summary || result.error || 'Completed'}</strong>`;

    document.getElementById('resultDetails').textContent = JSON.stringify(result, null, 2);

    document.getElementById('resultModal').classList.remove('hidden');
}

function closeModal() {
    document.getElementById('resultModal').classList.add('hidden');
}

// Close modal on outside click
document.getElementById('resultModal')?.addEventListener('click', (e) => {
    if (e.target.id === 'resultModal') closeModal();
});

// =============================================================================
// WEBSOCKET
// =============================================================================

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

    ws.onopen = () => {
        addTraceEntry('success', 'WebSocket connected');
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleWebSocketMessage(data);
    };

    ws.onclose = () => {
        addTraceEntry('warning', 'WebSocket disconnected');
        // Reconnect after 5 seconds
        setTimeout(connectWebSocket, 5000);
    };

    ws.onerror = (error) => {
        addTraceEntry('error', 'WebSocket error', error.message);
    };
}

function handleWebSocketMessage(data) {
    switch (data.type) {
        case 'tool_start':
            addTraceEntry('info', `Starting ${data.tool}`, JSON.stringify(data.params, null, 2));
            break;

        case 'tool_result':
            const status = data.result.error ? 'error' : 'success';
            addTraceEntry(status, `${data.tool} completed`, JSON.stringify(data.result, null, 2));
            if (data.result.workspace) {
                currentWorkspace = data.result.workspace;
                updateWorkspaceDisplay();
            }
            break;

        case 'scenario_start':
            addTraceEntry('info', `Starting scenario: ${data.name}`, `${data.steps} steps`);
            break;

        case 'step_start':
            addTraceEntry('info', `Step ${data.step}: ${data.tool}`);
            break;

        case 'step_result':
            const stepStatus = data.result.error ? 'warning' : 'success';
            addTraceEntry(stepStatus, `Step ${data.step} completed`, JSON.stringify(data.result, null, 2));
            break;

        case 'scenario_complete':
            addTraceEntry('success', `Scenario completed`, `Workspace: ${data.workspace}`);
            if (data.workspace) {
                currentWorkspace = data.workspace;
                updateWorkspaceDisplay();
            }
            break;
    }
}

// =============================================================================
// HISTORY SIDEBAR
// =============================================================================

function updateHistorySidebar() {
    const historyList = document.getElementById('historyList');
    if (!historyList) return;

    if (chatHistory.length === 0) {
        historyList.innerHTML = '<p class="history-empty">No history yet. Start a conversation!</p>';
        return;
    }

    historyList.innerHTML = '';
    chatHistory.forEach((entry, index) => {
        const item = document.createElement('div');
        item.className = `history-item ${entry.role}`;
        item.onclick = () => scrollToMessage(index);

        const icon = entry.role === 'user' ? '👤' : '🤖';
        item.innerHTML = `
            <span class="history-icon">${icon}</span>
            <span class="history-preview">${escapeHtml(entry.preview)}</span>
            <span class="history-time">${entry.timestamp}</span>
        `;
        historyList.appendChild(item);
    });

    // Scroll to bottom
    historyList.scrollTop = historyList.scrollHeight;
}

function scrollToMessage(index) {
    const messages = document.getElementById('chatMessages');
    const messageElements = messages.querySelectorAll('.message');
    if (messageElements[index]) {
        messageElements[index].scrollIntoView({ behavior: 'smooth', block: 'center' });
        messageElements[index].classList.add('highlight');
        setTimeout(() => messageElements[index].classList.remove('highlight'), 2000);
    }
}

// =============================================================================
// CHAT MANAGEMENT
// =============================================================================

function clearChat() {
    // Clear UI
    const messages = document.getElementById('chatMessages');
    messages.innerHTML = `
        <div class="message assistant">
            <div class="message-content">
                <p>Hello! I'm your KOAS assistant. I can help you with security scans and code audits. What would you like to do?</p>
            </div>
        </div>
    `;

    // Clear history
    chatHistory = [];
    updateHistorySidebar();

    // Clear server memory
    fetch('/api/memory', { method: 'DELETE' })
        .then(() => addTraceEntry('info', 'Chat and memory cleared'))
        .catch(err => console.error('Error clearing memory:', err));
}

async function loadMemory() {
    try {
        const response = await fetch('/api/memory');
        const data = await response.json();

        if (data.workspace) {
            currentWorkspace = data.workspace;
            updateWorkspaceDisplay();
        }
    } catch (error) {
        console.error('Error loading memory:', error);
    }
}

function updateWorkspaceDisplay() {
    const wsDisplay = document.getElementById('currentWorkspace');
    if (wsDisplay && currentWorkspace) {
        // Show shortened path
        const parts = currentWorkspace.split('/');
        const short = parts.slice(-2).join('/');
        wsDisplay.textContent = short;
        wsDisplay.title = currentWorkspace;
    }
}

// =============================================================================
// SESSION SAVE/LOAD
// =============================================================================

function saveSession() {
    sessionMode = 'save';
    document.getElementById('sessionModalTitle').textContent = 'Save Session';
    document.getElementById('sessionModalAction').textContent = 'Save';
    document.getElementById('sessionName').value = `session_${new Date().toISOString().slice(0, 10)}`;
    document.getElementById('savedSessions').style.display = 'none';
    document.getElementById('sessionModal').classList.remove('hidden');
}

async function loadSession() {
    sessionMode = 'load';
    document.getElementById('sessionModalTitle').textContent = 'Load Session';
    document.getElementById('sessionModalAction').textContent = 'Load';
    document.getElementById('sessionName').style.display = 'none';

    // Fetch saved sessions
    try {
        const response = await fetch('/api/memory/saved');
        const data = await response.json();

        const container = document.getElementById('savedSessions');
        container.style.display = 'block';
        container.innerHTML = '';

        if (data.sessions && data.sessions.length > 0) {
            data.sessions.forEach(session => {
                const item = document.createElement('div');
                item.className = 'saved-session-item';
                item.onclick = () => selectSession(session);
                item.innerHTML = `
                    <span class="session-name">${session}</span>
                `;
                container.appendChild(item);
            });
        } else {
            container.innerHTML = '<p class="no-sessions">No saved sessions found</p>';
        }
    } catch (error) {
        console.error('Error loading sessions:', error);
    }

    document.getElementById('sessionModal').classList.remove('hidden');
}

function selectSession(name) {
    document.querySelectorAll('.saved-session-item').forEach(el => el.classList.remove('selected'));
    event.target.closest('.saved-session-item').classList.add('selected');
    document.getElementById('sessionName').value = name;
}

async function confirmSessionAction() {
    const name = document.getElementById('sessionName').value.trim();
    if (!name) return;

    try {
        if (sessionMode === 'save') {
            const response = await fetch('/api/memory/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name })
            });
            const data = await response.json();
            addTraceEntry('success', 'Session saved', data.path);
        } else {
            const response = await fetch('/api/memory/load', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name })
            });
            const data = await response.json();

            if (data.success) {
                // Reload the chat with restored messages
                await loadMemory();
                addTraceEntry('success', 'Session loaded', name);
            }
        }
    } catch (error) {
        addTraceEntry('error', `Session ${sessionMode} failed`, error.message);
    }

    closeSessionModal();
}

function closeSessionModal() {
    document.getElementById('sessionModal').classList.add('hidden');
    document.getElementById('sessionName').style.display = 'block';
}

// =============================================================================
// UTILITIES
// =============================================================================

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Category filter change handler
document.getElementById('categoryFilter')?.addEventListener('change', renderTools);
