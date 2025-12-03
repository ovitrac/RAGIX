/**
 * RAGIX Web UI - Main Application
 *
 * Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-24
 */

class RAGIXApp {
    constructor() {
        this.ws = null;
        this.sessionId = 'default';
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.messageHistory = [];

        this.init();
    }

    init() {
        // DOM elements
        this.chatMessages = document.getElementById('chatMessages');
        this.chatInput = document.getElementById('chatInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.statusEl = document.getElementById('status');
        this.settingsBtn = document.getElementById('settingsBtn');
        this.settingsModal = document.getElementById('settingsModal');
        this.tracePanel = document.getElementById('tracePanel');

        // Event listeners
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        this.chatInput.addEventListener('keydown', (e) => this.handleInputKeydown(e));
        this.settingsBtn.addEventListener('click', () => this.openSettings());

        // Connect WebSocket
        this.connect();

        // Load session info
        this.loadSessionInfo();

        // Show welcome message
        this.addSystemMessage('Welcome to RAGIX Web UI! Type a message to start.');
    }

    connect() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/chat/${this.sessionId}`;

        this.updateStatus('connecting');

        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.updateStatus('connected');
            this.reconnectAttempts = 0;
            // Refresh session info now that agent is initialized
            setTimeout(() => this.loadSessionInfo(), 500);
        };

        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleMessage(data);
            } catch (error) {
                console.error('Failed to parse message:', error);
            }
        };

        this.ws.onclose = () => {
            console.log('WebSocket closed');
            this.updateStatus('disconnected');
            this.attemptReconnect();
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.updateStatus('error');
        };
    }

    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`Reconnecting... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
            setTimeout(() => this.connect(), 2000 * this.reconnectAttempts);
        } else {
            this.addSystemMessage('Connection lost. Please refresh the page.', 'error');
        }
    }

    handleMessage(data) {
        const { type, message, timestamp } = data;

        switch (type) {
            case 'status':
                this.addSystemMessage(message);
                break;

            case 'user_message':
                this.addUserMessage(message, timestamp);
                break;

            case 'thinking':
                this.showThinking(message);
                break;

            case 'agent_message':
                this.hideThinking();
                this.addAgentMessage(message, timestamp);
                break;

            case 'tool_call':
                this.addToolTrace(data);
                break;

            case 'reasoning_traces':
                // Handle reasoning traces from Planner/Worker/Verifier loop
                if (data.traces && Array.isArray(data.traces)) {
                    this.addReasoningTraces(data.traces);
                }
                break;

            case 'reasoning_graph_state':
                // v0.23: Handle reasoning graph state updates
                if (typeof reasoningGraph !== 'undefined') {
                    reasoningGraph.updateFromWebSocket(data);
                }
                break;

            case 'progress':
                // Handle streaming progress updates
                if (data.event) {
                    this.handleProgressUpdate(data.event);
                }
                break;

            case 'reasoning_trace_update':
                // Handle real-time trace updates (add to trace panel immediately)
                if (data.trace) {
                    this.addSingleReasoningTrace(data.trace);
                }
                break;

            case 'error':
                this.hideThinking();
                this.addSystemMessage(message, 'error');
                break;

            case 'pong':
                // Heartbeat response
                break;

            default:
                console.warn('Unknown message type:', type);
        }
    }

    showThinking(message = 'Agent is processing...') {
        // Remove any existing thinking indicator
        this.hideThinking();

        const thinkingEl = document.createElement('div');
        thinkingEl.id = 'thinking-indicator';
        thinkingEl.className = 'message message-system message-thinking';
        thinkingEl.innerHTML = `
            <div class="message-content">
                <span class="thinking-dots">${this.escapeHtml(message)}</span>
                <span class="thinking-spinner"></span>
            </div>
        `;
        this.chatMessages.appendChild(thinkingEl);
        this.scrollToBottom();
    }

    hideThinking() {
        const thinkingEl = document.getElementById('thinking-indicator');
        if (thinkingEl) {
            thinkingEl.remove();
        }
    }

    updateThinking(message) {
        // Update the thinking indicator text without removing it
        const thinkingEl = document.getElementById('thinking-indicator');
        if (thinkingEl) {
            const dotsEl = thinkingEl.querySelector('.thinking-dots');
            if (dotsEl) {
                dotsEl.textContent = message;
            }
        } else {
            this.showThinking(message);
        }
    }

    handleProgressUpdate(event) {
        // Update thinking indicator with progress info
        const eventType = event.type || 'processing';
        const content = event.content || '';
        const elapsed = event.elapsed ? `[${event.elapsed.toFixed(1)}s]` : '';

        // Format progress message based on event type
        let progressMsg = `${elapsed} ${content}`.trim();

        switch (eventType) {
            case 'classification':
            case 'classification_complete':
                progressMsg = `${elapsed} 📊 ${content}`;
                break;
            case 'graph_start':
                progressMsg = `${elapsed} 🚀 Starting reasoning...`;
                break;
            case 'planning':
                progressMsg = `${elapsed} 📝 Planning steps...`;
                break;
            case 'plan_ready':
                progressMsg = `${elapsed} 📋 ${content}`;
                break;
            case 'plan_step':
                progressMsg = `${elapsed} ${content}`;
                break;
            case 'executing':
            case 'execution':
            case 'step_execution':
                progressMsg = `${elapsed} ⚙️ ${content || 'Executing...'}`;
                break;
            case 'step_complete':
                progressMsg = `${elapsed} ${content}`;
                break;
            case 'reflection':
                progressMsg = `${elapsed} 🔄 Reflecting on results...`;
                break;
            case 'verification':
                progressMsg = `${elapsed} 🔍 Verifying results...`;
                break;
            case 'responding':
                progressMsg = `${elapsed} 💬 Preparing response...`;
                break;
            case 'complete':
                progressMsg = `${elapsed} ✅ ${content}`;
                break;
            case 'error':
                progressMsg = `${elapsed} ❌ ${content}`;
                break;
            default:
                if (content) {
                    progressMsg = `${elapsed} ${content}`;
                }
        }

        this.updateThinking(progressMsg || 'Processing...');
    }

    addSingleReasoningTrace(trace) {
        // Add a single trace to the trace panel in real-time
        const traceContent = document.getElementById('traceContent');
        if (!traceContent) return;

        // Remove empty message if present
        const emptyMsg = traceContent.querySelector('.trace-empty');
        if (emptyMsg) emptyMsg.remove();

        const traceType = trace.type || 'trace';
        const content = trace.content || '';
        const elapsed = trace.elapsed ? `${trace.elapsed.toFixed(1)}s` : '';

        // Icon mapping for trace types
        const icons = {
            'classification': '📊',
            'classification_complete': '📊',
            'graph_start': '🚀',
            'graph_complete': '🏁',
            'planning': '📝',
            'plan_ready': '📋',
            'plan_step': '📌',
            'execution': '⚙️',
            'step_execution': '⚙️',
            'executing': '⚙️',
            'step_complete': '✅',
            'direct_execution': '⚡',
            'reflection': '🔄',
            'verification': '🔍',
            'responding': '💬',
            'complete': '✅',
            'timeout': '⏱️',
            'error': '❌',
            'bypass': '💬'
        };
        const icon = icons[traceType] || '📌';

        const traceEl = document.createElement('div');
        traceEl.className = `trace-item trace-${traceType}`;
        traceEl.innerHTML = `
            <div class="trace-header">
                <span class="trace-icon">${icon}</span>
                <span class="trace-type">${traceType}</span>
                <span class="trace-elapsed">${elapsed}</span>
            </div>
            <div class="trace-content">${this.escapeHtml(content.substring(0, 200))}${content.length > 200 ? '...' : ''}</div>
        `;

        traceContent.appendChild(traceEl);

        // Auto-scroll trace panel to bottom
        traceContent.scrollTop = traceContent.scrollHeight;
    }

    sendMessage() {
        const message = this.chatInput.value.trim();
        if (!message) return;

        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            this.addSystemMessage('Not connected. Please wait...', 'error');
            return;
        }

        // Send to server
        this.ws.send(JSON.stringify({
            type: 'chat',
            message: message
        }));

        // Clear input
        this.chatInput.value = '';
        this.chatInput.style.height = 'auto';
    }

    handleInputKeydown(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            this.sendMessage();
        }
    }

    addUserMessage(message, timestamp) {
        const messageEl = document.createElement('div');
        messageEl.className = 'message message-user';
        messageEl.innerHTML = `
            <div class="message-header">
                <span class="message-sender">You</span>
                <span class="message-time">${this.formatTime(timestamp)}</span>
            </div>
            <div class="message-content">${this.escapeHtml(message)}</div>
        `;
        this.chatMessages.appendChild(messageEl);
        this.scrollToBottom();
    }

    addAgentMessage(message, timestamp) {
        const messageEl = document.createElement('div');
        messageEl.className = 'message message-agent';
        messageEl.innerHTML = `
            <div class="message-header">
                <span class="message-sender">RAGIX Agent</span>
                <span class="message-time">${this.formatTime(timestamp)}</span>
            </div>
            <div class="message-content">${this.formatMessage(message)}</div>
        `;
        this.chatMessages.appendChild(messageEl);
        this.scrollToBottom();
    }

    addSystemMessage(message, level = 'info') {
        const messageEl = document.createElement('div');
        messageEl.className = `message message-system message-${level}`;
        messageEl.innerHTML = `
            <div class="message-content">${this.escapeHtml(message)}</div>
        `;
        this.chatMessages.appendChild(messageEl);
        this.scrollToBottom();
    }

    addToolTrace(data) {
        // Add to trace panel
        const traceEl = document.createElement('div');
        traceEl.className = 'trace-item';
        traceEl.innerHTML = `
            <div class="trace-header">
                <span class="trace-tool">${data.tool || 'unknown'}</span>
                <span class="trace-time">${this.formatTime(data.timestamp)}</span>
            </div>
            <div class="trace-body">
                <pre>${this.escapeHtml(JSON.stringify(data, null, 2))}</pre>
            </div>
        `;

        const traceContent = document.getElementById('traceContent');
        const emptyMsg = traceContent.querySelector('.trace-empty');
        if (emptyMsg) emptyMsg.remove();

        traceContent.appendChild(traceEl);
    }

    addReasoningTraces(traces) {
        // Add reasoning traces to the Reasoning tab panel
        const traceContent = document.getElementById('traceContent');
        if (!traceContent) return;

        // Remove empty message if present
        const emptyMsg = traceContent.querySelector('.trace-empty');
        if (emptyMsg) emptyMsg.remove();

        // Agent icons for different types
        const agentIcons = {
            'planner': '&#x1F4CB;',      // clipboard
            'planning': '&#x1F4CB;',
            'worker': '&#x1F6E0;',       // wrench
            'execution': '&#x1F6E0;',
            'step_completed': '&#x2705;', // check mark
            'step_failed': '&#x274C;',    // cross mark
            'verifier': '&#x2714;',      // check
            'verification': '&#x2714;',
            'verify_complete': '&#x2714;',
            'classification': '&#x1F50D;', // magnifying glass
            'plan_generated': '&#x1F4DD;', // memo
            'direct_execution': '&#x26A1;', // lightning
            'execution_halted': '&#x26D4;', // stop
        };

        traces.forEach(trace => {
            const traceEl = document.createElement('div');
            traceEl.className = 'trace-item reasoning-trace';

            // Determine agent type from trace
            const traceType = trace.type || trace.agent || 'unknown';
            const icon = agentIcons[traceType] || '&#x1F916;'; // robot default
            const timestamp = trace.timestamp ? this.formatTime(trace.timestamp) : '';

            // Format content - handle both trace structures
            let content = trace.content || trace.message || JSON.stringify(trace);

            // Truncate very long content
            const maxLength = 500;
            if (content.length > maxLength) {
                content = content.substring(0, maxLength) + '...';
            }

            traceEl.innerHTML = `
                <div class="trace-header">
                    <span class="trace-agent">${icon} ${this.escapeHtml(traceType)}</span>
                    <span class="trace-time">${timestamp}</span>
                </div>
                <div class="trace-body">
                    <pre>${this.escapeHtml(content)}</pre>
                </div>
            `;

            traceContent.appendChild(traceEl);
        });

        // Scroll to bottom of trace panel
        traceContent.scrollTop = traceContent.scrollHeight;
    }

    updateStatus(status) {
        this.statusEl.className = `status status-${status}`;
        const statusText = {
            'connecting': 'Connecting...',
            'connected': 'Connected',
            'disconnected': 'Disconnected',
            'error': 'Error'
        };
        this.statusEl.textContent = statusText[status] || status;
    }

    async loadSessionInfo() {
        try {
            // Use the new status endpoint that returns actual values
            const response = await fetch(`/api/sessions/${encodeURIComponent(this.sessionId)}/status`);

            if (response.ok) {
                const data = await response.json();
                document.getElementById('sandboxPath').textContent = data.sandbox_root || '-';
                document.getElementById('modelName').textContent = data.model || '-';
                document.getElementById('profileName').textContent = data.profile || '-';
                document.getElementById('reasoningStrategy').textContent = data.reasoning_strategy || '-';
            } else {
                // Fallback to basic session list if status endpoint fails
                const listResp = await fetch(`/api/sessions`);
                const listData = await listResp.json();
                const session = listData.sessions.find(s => s.id === this.sessionId);
                if (session) {
                    document.getElementById('sandboxPath').textContent = session.sandbox_root || '-';
                    document.getElementById('modelName').textContent = session.model || '-';
                    document.getElementById('profileName').textContent = session.profile || '-';
                }

                // Fetch reasoning separately
                try {
                    const reasoningResp = await fetch(`/api/reasoning/status?session_id=${encodeURIComponent(this.sessionId)}`);
                    const reasoningData = await reasoningResp.json();
                    document.getElementById('reasoningStrategy').textContent = reasoningData.strategy || '-';
                } catch (e) {
                    console.error('Failed to load reasoning status:', e);
                }
            }
        } catch (error) {
            console.error('Failed to load session info:', error);
        }
    }

    async openSettings() {
        this.settingsModal.classList.remove('hidden');

        // Always update session ID field to current session
        document.getElementById('sessionIdInput').value = this.sessionId;

        // Load current session values
        try {
            const response = await fetch(`/api/sessions/${encodeURIComponent(this.sessionId)}/status`);
            if (response.ok) {
                const data = await response.json();
                // Store current session model to select after loading models
                this._currentSessionModel = data.model || data.session_model || 'mistral';
                this._currentSessionSandbox = data.sandbox_root || '';
                this._currentSessionProfile = data.profile || 'dev';

                // Update form fields with current values
                document.getElementById('sandboxInput').value = this._currentSessionSandbox;
                document.getElementById('profileInput').value = this._currentSessionProfile;
            } else {
                // Use defaults if status fetch fails
                this._currentSessionModel = 'mistral';
            }
        } catch (e) {
            console.error('Failed to load session status:', e);
            this._currentSessionModel = 'mistral';
        }

        // Load models and select the current one
        await this.loadOllamaModels();
    }

    closeSettings() {
        this.settingsModal.classList.add('hidden');
    }

    async loadOllamaModels() {
        const select = document.getElementById('modelInput');
        const statusEl = document.getElementById('modelStatus');

        try {
            select.innerHTML = '<option value="">Loading models...</option>';
            statusEl.textContent = 'Fetching available models from Ollama...';

            const response = await fetch('/api/ollama/models');
            const data = await response.json();

            if (!data.available) {
                select.innerHTML = '<option value="mistral">mistral (default)</option>';
                statusEl.textContent = `Ollama not available: ${data.error}`;
                statusEl.style.color = 'var(--error)';
                return;
            }

            if (data.models.length === 0) {
                select.innerHTML = '<option value="">No models installed</option>';
                statusEl.textContent = 'No models found. Run: ollama pull mistral';
                statusEl.style.color = 'var(--warning)';
                return;
            }

            // Build options from available models
            const currentModel = this._currentSessionModel || 'mistral';
            select.innerHTML = data.models.map(m => {
                const isSelected = m.name === currentModel || m.name.startsWith(currentModel.split(':')[0]);
                return `<option value="${m.name}" ${isSelected ? 'selected' : ''}>${m.name} (${m.size})</option>`;
            }).join('');

            // If current model not found in list, try exact match
            if (select.value !== currentModel) {
                for (const opt of select.options) {
                    if (opt.value === currentModel) {
                        opt.selected = true;
                        break;
                    }
                }
            }

            statusEl.textContent = `${data.count} sovereign AI model(s) available locally`;
            statusEl.style.color = 'var(--success)';

        } catch (error) {
            console.error('Failed to load Ollama models:', error);
            select.innerHTML = '<option value="mistral">mistral (fallback)</option>';
            statusEl.textContent = 'Failed to fetch models. Using default.';
            statusEl.style.color = 'var(--error)';
        }
    }

    async saveSettings() {
        const sessionId = document.getElementById('sessionIdInput').value;
        const sandbox = document.getElementById('sandboxInput').value;
        const model = document.getElementById('modelInput').value;
        const profile = document.getElementById('profileInput').value;

        try {
            // Create new session
            const response = await fetch('/api/sessions', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    sandbox_root: sandbox,
                    model: model,
                    profile: profile
                })
            });

            const data = await response.json();
            this.sessionId = data.session_id;

            // Reconnect
            if (this.ws) {
                this.ws.close();
            }
            this.connect();
            this.loadSessionInfo();
            this.closeSettings();

            this.addSystemMessage('Settings updated. Reconnected.');
        } catch (error) {
            console.error('Failed to save settings:', error);
            this.addSystemMessage('Failed to update settings.', 'error');
        }
    }

    showLogs() {
        // TODO: Implement logs viewer
        this.addSystemMessage('Logs viewer coming soon...');
    }

    showTraces() {
        this.tracePanel.classList.toggle('hidden');
    }

    closeTraces() {
        this.tracePanel.classList.add('hidden');
    }

    clearChat() {
        if (confirm('Clear all messages?')) {
            this.chatMessages.innerHTML = '';
            this.addSystemMessage('Chat cleared.');
        }
    }

    formatTime(timestamp) {
        if (!timestamp) return '';
        const date = new Date(timestamp);
        return date.toLocaleTimeString();
    }

    formatMessage(text) {
        // Simple markdown-like formatting
        text = this.escapeHtml(text);

        // Extract JSON section first (before other processing)
        let jsonContent = null;
        const jsonMatch = text.match(/&lt;!-- JSON_START --&gt;([\s\S]*?)&lt;!-- JSON_END --&gt;/);
        if (jsonMatch) {
            jsonContent = jsonMatch[1].trim();
            // Remove JSON section from visible text
            text = text.replace(/&lt;!-- JSON_START --&gt;[\s\S]*?&lt;!-- JSON_END --&gt;/, '');
        }

        // Handle collapsible details section with copy buttons
        const detailsMatch = text.match(/&lt;!-- DETAILS_START --&gt;([\s\S]*?)&lt;!-- DETAILS_END --&gt;/);
        if (detailsMatch) {
            const detailsContent = detailsMatch[1].trim();
            const detailsId = 'details-' + Date.now();

            // Build action buttons (always Copy, optionally Copy JSON)
            let actionButtons = `<button onclick="copyToClipboard('${detailsId}-text')" class="copy-btn" title="Copy output">&#128203; Copy</button>`;
            if (jsonContent) {
                actionButtons += `<button onclick="copyToClipboard('${detailsId}-json')" class="copy-btn copy-json-btn" title="Copy as JSON">&#123;&#125; JSON</button>`;
            }

            // Hidden JSON element for copy
            const jsonElement = jsonContent ? `<pre id="${detailsId}-json" style="display:none;">${jsonContent}</pre>` : '';

            const detailsHtml = `
                <div class="details-section">
                    <div class="details-toggle" onclick="toggleDetails('${detailsId}')">
                        <span class="toggle-icon">&#9654;</span> Show details
                    </div>
                    <div id="${detailsId}" class="details-content" style="display:none;">
                        <div class="details-actions">
                            ${actionButtons}
                        </div>
                        <pre id="${detailsId}-text">${detailsContent}</pre>
                        ${jsonElement}
                    </div>
                </div>`;
            text = text.replace(/&lt;!-- DETAILS_START --&gt;[\s\S]*?&lt;!-- DETAILS_END --&gt;/, detailsHtml);
        } else if (jsonContent) {
            // JSON without details section - add standalone JSON copy button
            const jsonId = 'json-' + Date.now();
            const jsonHtml = `
                <div class="json-copy-section">
                    <button onclick="copyToClipboard('${jsonId}')" class="copy-btn copy-json-btn" title="Copy as JSON">&#123;&#125; Copy JSON</button>
                    <pre id="${jsonId}" style="display:none;">${jsonContent}</pre>
                </div>`;
            text += jsonHtml;
        }

        // Code blocks with language hint and copy button
        let codeBlockId = 0;
        text = text.replace(/```(\w+)?\n([\s\S]*?)```/g, (match, lang, code) => {
            const blockId = `code-block-${Date.now()}-${codeBlockId++}`;
            return `<div class="code-block">
                <div class="code-header">
                    <span class="code-lang">${lang || 'output'}</span>
                    <button class="code-copy-btn" onclick="copyCodeBlock('${blockId}')" title="Copy code">&#128203;</button>
                </div>
                <pre><code id="${blockId}">${code}</code></pre>
            </div>`;
        });

        // Inline code
        text = text.replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>');

        // Bold
        text = text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

        // Italic
        text = text.replace(/\*([^*]+)\*/g, '<em>$1</em>');

        // Command output section detection (for agent responses)
        text = text.replace(/\$ ([^\n]+)\n/g, '<div class="cmd-line">$ $1</div>');

        // Output sections
        text = text.replace(/Output:\n/g, '<div class="output-header">Output:</div>');

        // Return code
        text = text.replace(/Return code: (\d+)/g, (match, code) => {
            const statusClass = code === '0' ? 'success' : 'error';
            return `<div class="return-code ${statusClass}">Return code: ${code}</div>`;
        });

        // Line breaks
        text = text.replace(/\n/g, '<br>');

        return text;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    scrollToBottom() {
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }

    // =========================================================================
    // Quick Actions & Demo Prompts
    // =========================================================================

    /**
     * Load a quick action prompt by icon name
     * @param {string} iconName - The icon identifier (folder, chart, clock, bug, link, heart)
     */
    async loadQuickAction(iconName) {
        try {
            const response = await fetch('/api/prompts/quick-actions');
            if (!response.ok) {
                throw new Error('Failed to fetch quick actions');
            }

            const data = await response.json();
            const prompts = data.prompts || [];

            // Map icon names to prompt icons
            const iconMapping = {
                'folder': 'folder',
                'chart': 'chart',
                'clock': 'clock',
                'bug': 'bug',
                'link': 'link',
                'heart': 'heart'
            };

            // Find matching prompt
            const targetIcon = iconMapping[iconName];
            const prompt = prompts.find(p => p.icon === targetIcon);

            if (prompt) {
                this.setPromptAndSend(prompt.prompt);
            } else {
                // Fallback prompts if API doesn't return matching icon
                const fallbackPrompts = {
                    'folder': 'Give me an overview of this project: main directories, key files, and purpose.',
                    'chart': 'Generate code statistics: total files, lines of code, file types distribution.',
                    'clock': 'What files were modified most recently? Show the last 10 changed files.',
                    'bug': 'Search for common bug patterns: unhandled exceptions, null checks, resource leaks.',
                    'link': 'List all external dependencies from requirements.txt or pyproject.toml.',
                    'heart': 'Perform a project health check: verify all imports work, no syntax errors, tests pass.'
                };

                if (fallbackPrompts[iconName]) {
                    this.setPromptAndSend(fallbackPrompts[iconName]);
                }
            }
        } catch (error) {
            console.error('Failed to load quick action:', error);
            // Use fallback directly on error
            const fallbackPrompts = {
                'folder': 'Give me an overview of this project: main directories, key files, and purpose.',
                'chart': 'Generate code statistics: total files, lines of code, file types distribution.',
                'clock': 'What files were modified most recently? Show the last 10 changed files.',
                'bug': 'Search for common bug patterns: unhandled exceptions, null checks, resource leaks.',
                'link': 'List all external dependencies from requirements.txt or pyproject.toml.',
                'heart': 'Perform a project health check: verify all imports work, no syntax errors, tests pass.'
            };

            if (fallbackPrompts[iconName]) {
                this.setPromptAndSend(fallbackPrompts[iconName]);
            }
        }
    }

    /**
     * Load prompts by complexity level and display in dropdown list
     * @param {string} complexity - The complexity level (bypass, simple, moderate, complex)
     */
    async loadPromptsByComplexity(complexity) {
        const promptsList = document.getElementById('promptsList');
        const select = document.getElementById('promptComplexity');

        if (!complexity) {
            promptsList.classList.add('hidden');
            return;
        }

        promptsList.classList.remove('hidden');
        promptsList.innerHTML = '<div class="prompt-loading">Loading prompts...</div>';

        try {
            const response = await fetch(`/api/prompts?complexity=${encodeURIComponent(complexity)}`);
            if (!response.ok) {
                throw new Error('Failed to fetch prompts');
            }

            const data = await response.json();
            const prompts = data.prompts || [];

            if (prompts.length === 0) {
                promptsList.innerHTML = '<div class="prompt-empty">No prompts found for this complexity level.</div>';
                return;
            }

            promptsList.innerHTML = prompts.map(p => `
                <div class="prompt-item" onclick="app.selectPrompt('${this.escapeHtml(p.prompt.replace(/'/g, "\\'"))}')">
                    <span class="prompt-item-name">${this.escapeHtml(p.name)}</span>
                    <span class="prompt-item-category">${this.escapeHtml(p.category || 'general')}</span>
                </div>
            `).join('');

        } catch (error) {
            console.error('Failed to load prompts:', error);
            promptsList.innerHTML = '<div class="prompt-empty">Failed to load prompts. Check API connection.</div>';
        }
    }

    /**
     * Select a prompt from the list and set it in the input
     * @param {string} promptText - The prompt text to set
     */
    selectPrompt(promptText) {
        // Hide the prompts list
        const promptsList = document.getElementById('promptsList');
        promptsList.classList.add('hidden');

        // Reset the complexity selector
        const select = document.getElementById('promptComplexity');
        select.value = '';

        // Set the prompt in the input
        this.chatInput.value = promptText;
        this.chatInput.focus();
    }

    /**
     * Set prompt text in input and optionally send immediately
     * @param {string} promptText - The prompt text
     * @param {boolean} autoSend - Whether to send immediately (default: true)
     */
    setPromptAndSend(promptText, autoSend = true) {
        this.chatInput.value = promptText;

        if (autoSend) {
            this.sendMessage();
        } else {
            this.chatInput.focus();
        }
    }
}

// Initialize app when DOM is ready
const app = new RAGIXApp();

// Heartbeat to keep connection alive
setInterval(() => {
    if (app.ws && app.ws.readyState === WebSocket.OPEN) {
        app.ws.send(JSON.stringify({ type: 'ping' }));
    }
}, 30000);

// Global helper functions for collapsible details
function toggleDetails(id) {
    const content = document.getElementById(id);
    const toggle = content.previousElementSibling;
    const icon = toggle.querySelector('.toggle-icon');

    if (content.style.display === 'none') {
        content.style.display = 'block';
        icon.innerHTML = '&#9660;'; // down arrow
        toggle.innerHTML = toggle.innerHTML.replace('Show details', 'Hide details');
    } else {
        content.style.display = 'none';
        icon.innerHTML = '&#9654;'; // right arrow
        toggle.innerHTML = toggle.innerHTML.replace('Hide details', 'Show details');
    }
}

function copyToClipboard(id) {
    const element = document.getElementById(id);
    const text = element.textContent || element.innerText;

    navigator.clipboard.writeText(text).then(() => {
        // Show feedback
        const btn = event.target;
        const originalText = btn.innerHTML;
        btn.innerHTML = '&#10003; Copied!';
        btn.classList.add('copied');
        setTimeout(() => {
            btn.innerHTML = originalText;
            btn.classList.remove('copied');
        }, 2000);
    }).catch(err => {
        console.error('Copy failed:', err);
        // Fallback for older browsers
        const textarea = document.createElement('textarea');
        textarea.value = text;
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
    });
}

function copyCodeBlock(id) {
    const element = document.getElementById(id);
    if (!element) return;

    // Get text content, handling HTML entities
    const text = element.textContent || element.innerText;

    navigator.clipboard.writeText(text).then(() => {
        // Find the copy button in the same code block
        const codeBlock = element.closest('.code-block');
        const btn = codeBlock ? codeBlock.querySelector('.code-copy-btn') : null;

        if (btn) {
            const originalText = btn.innerHTML;
            btn.innerHTML = '&#10003;';
            btn.classList.add('copied');
            setTimeout(() => {
                btn.innerHTML = originalText;
                btn.classList.remove('copied');
            }, 2000);
        }
    }).catch(err => {
        console.error('Copy code block failed:', err);
        // Fallback
        const textarea = document.createElement('textarea');
        textarea.value = text;
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
    });
}
