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

        // File handling
        this.pendingFiles = [];
        this.setupFileHandling();

        // Memory explorer
        this.setupMemorySearch();

        // Connect WebSocket
        this.connect();

        // Load session info
        this.loadSessionInfo();

        // Load app version from server
        this.loadAppVersion();

        // Load threads (v0.33)
        this.loadThreads();

        // Load RAG status (v0.33)
        this.loadRagStatus();

        // Initialize global context editor (v0.33)
        this.initGlobalContext();

        // Show welcome message
        this.addSystemMessage('Welcome to RAGIX Web UI! Type a message to start.');
    }

    async loadAppVersion() {
        try {
            const response = await fetch('/api/health');
            if (response.ok) {
                const data = await response.json();
                const versionEl = document.getElementById('app-version');
                if (versionEl && data.version) {
                    versionEl.textContent = `v${data.version}`;
                }
            }
        } catch (error) {
            console.log('Could not load app version:', error);
        }
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
                this.showThinking(message, data.cancellable !== false);
                break;

            case 'rag_context':
                // v0.33: Show RAG retrieval notification
                this.addSystemMessage(message, 'info');
                break;

            case 'cancel_ack':
                this.updateThinking('⛔ Cancellation requested...');
                break;

            case 'agent_message':
                this.hideThinking();
                this.addAgentMessage(message, timestamp, data.token_stats);
                // Force refresh model info to get actual VRAM (model is now loaded)
                // Clear cache to force re-fetch with ?refresh=true
                this._modelInfoCache = null;
                // Update context window after receiving response (also refreshes model info)
                this.updateContextWindow();
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
                // v0.32: Update memory context with reasoning state
                if (typeof memoryContext !== 'undefined') {
                    memoryContext.updateFromWebSocket(data);
                }
                break;

            case 'progress':
                // Handle streaming progress updates
                if (data.event) {
                    this.handleProgressUpdate(data.event);
                    // Update context window periodically during reasoning
                    // Throttle to avoid excessive API calls (every 5 seconds)
                    const now = Date.now();
                    if (!this._lastContextUpdate || now - this._lastContextUpdate > 5000) {
                        this._lastContextUpdate = now;
                        this.updateContextWindow();
                    }
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

    showThinking(message = 'Agent is processing...', cancellable = true) {
        // Remove any existing thinking indicator
        this.hideThinking();

        const thinkingEl = document.createElement('div');
        thinkingEl.id = 'thinking-indicator';
        thinkingEl.className = 'message message-system message-thinking';

        const cancelButton = cancellable ? `
            <button class="cancel-btn" onclick="app.cancelRequest()" title="Stop reasoning">
                ⛔ Stop
            </button>
        ` : '';

        thinkingEl.innerHTML = `
            <div class="message-content">
                <span class="thinking-dots">${this.escapeHtml(message)}</span>
                <span class="thinking-spinner"></span>
                ${cancelButton}
            </div>
        `;
        this.chatMessages.appendChild(thinkingEl);
        this.scrollToBottom();
    }

    cancelRequest() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'cancel' }));
            this.updateThinking('⛔ Cancelling...');
            // Disable the cancel button
            const cancelBtn = document.querySelector('.cancel-btn');
            if (cancelBtn) {
                cancelBtn.disabled = true;
                cancelBtn.textContent = 'Cancelling...';
            }
        }
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
        const metadata = event.metadata || event;

        // Format progress message based on event type
        let progressMsg = `${elapsed} ${content}`.trim();

        switch (eventType) {
            case 'classification':
            case 'classification_complete':
                progressMsg = `${elapsed} 📊 ${content}`;
                // Show classification as inline card
                this.addProgressCard('classification', content, elapsed, metadata);
                break;
            case 'graph_start':
                progressMsg = `${elapsed} 🚀 Starting reasoning...`;
                break;
            case 'planning':
                progressMsg = `${elapsed} 📝 Planning steps...`;
                break;
            case 'plan_ready':
                progressMsg = `${elapsed} 📋 ${content}`;
                // Show plan as inline card
                this.addProgressCard('plan', content, elapsed, metadata);
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
                // Show step result as inline card
                this.addProgressCard('step', content, elapsed, metadata);
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
                this.addProgressCard('error', content, elapsed, metadata);
                break;
            case 'cancelled':
                progressMsg = `${elapsed} ⛔ ${content}`;
                this.addProgressCard('cancelled', content, elapsed, metadata);
                break;
            default:
                if (content) {
                    progressMsg = `${elapsed} ${content}`;
                }
        }

        this.updateThinking(progressMsg || 'Processing...');
    }

    addProgressCard(type, content, elapsed, metadata = {}) {
        // Add an inline progress card to the chat area
        const cardEl = document.createElement('div');
        cardEl.className = `progress-card progress-card-${type}`;

        // Determine icon and status class
        const icons = {
            'classification': '📊',
            'plan': '📋',
            'step': '⚙️',
            'error': '❌',
            'cancelled': '⛔'
        };
        const icon = icons[type] || '📌';

        // Determine status badge (info = no badge, just informational)
        let statusBadge = '';
        const status = metadata.status || (type === 'error' ? 'failed' : 'success');
        if (status === 'success') {
            statusBadge = '<span class="status-badge status-success">✅ Success</span>';
        } else if (status === 'failed') {
            statusBadge = '<span class="status-badge status-error">❌ Failed</span>';
        } else if (status === 'cancelled') {
            statusBadge = '<span class="status-badge status-warning">⛔ Cancelled</span>';
        }
        // status === 'info' → no badge (informational only)

        // Build result preview if available
        let resultPreview = '';
        if (metadata.result && metadata.result.trim()) {
            const preview = metadata.result.substring(0, 200);
            const truncated = metadata.result.length > 200 ? '...' : '';
            resultPreview = `
                <details class="result-details">
                    <summary>View output</summary>
                    <pre class="result-preview">${this.escapeHtml(preview)}${truncated}</pre>
                </details>
            `;
        }

        // Build step number if available
        const stepNum = metadata.step_num ? `Step ${metadata.step_num}` : '';

        cardEl.innerHTML = `
            <div class="progress-card-header">
                <span class="progress-card-icon">${icon}</span>
                <span class="progress-card-title">${stepNum || type}</span>
                <span class="progress-card-elapsed">${elapsed}</span>
                ${statusBadge}
            </div>
            <div class="progress-card-content">${this.escapeHtml(content.substring(0, 150))}</div>
            ${resultPreview}
        `;

        // Insert before the thinking indicator
        const thinkingEl = document.getElementById('thinking-indicator');
        if (thinkingEl) {
            this.chatMessages.insertBefore(cardEl, thinkingEl);
        } else {
            this.chatMessages.appendChild(cardEl);
        }
        this.scrollToBottom();
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
            'cancelled': '⛔',
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

        // Get file context if files are attached
        const fileContext = this.getFileContextForMessage();

        // v0.33: Get global context
        const globalContext = this.getGlobalContext();

        // Build full message with context
        let fullMessage = message;

        // Add global context if present
        if (globalContext) {
            fullMessage = `[USER INSTRUCTIONS]\n${globalContext}\n\n[REQUEST]\n${message}`;
        }

        // Add file context if present
        if (fileContext) {
            if (globalContext) {
                fullMessage = `[USER INSTRUCTIONS]\n${globalContext}\n\n[ATTACHED FILES]\n${fileContext}\n\n[REQUEST]\n${message}`;
            } else {
                fullMessage = `[ATTACHED FILES]\n${fileContext}\n\n[REQUEST]\n${message}`;
            }
        }

        // Send to server
        this.ws.send(JSON.stringify({
            type: 'chat',
            message: fullMessage
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

    addAgentMessage(message, timestamp, tokenStats = null) {
        const messageEl = document.createElement('div');
        messageEl.className = 'message message-agent';

        // Format token stats if available
        let tokenStatsHtml = '';
        if (tokenStats && tokenStats.total_tokens > 0) {
            const promptTok = this.formatTokenCount(tokenStats.total_prompt_tokens);
            const completionTok = this.formatTokenCount(tokenStats.total_completion_tokens);
            const totalTok = this.formatTokenCount(tokenStats.total_tokens);
            const reqCount = tokenStats.request_count || 1;

            // Last request stats
            const lastReq = tokenStats.last_request || {};
            const tps = lastReq.tokens_per_second ? lastReq.tokens_per_second.toFixed(1) : '?';

            tokenStatsHtml = `
                <div class="token-stats">
                    <span class="token-stat" title="Prompt tokens (input)">↑${promptTok}</span>
                    <span class="token-stat" title="Completion tokens (output)">↓${completionTok}</span>
                    <span class="token-stat token-total" title="Total tokens this session">Σ${totalTok}</span>
                    <span class="token-stat" title="LLM requests">${reqCount} req</span>
                    <span class="token-stat" title="Tokens per second">${tps} tok/s</span>
                </div>
            `;
        }

        messageEl.innerHTML = `
            <div class="message-header">
                <span class="message-sender">RAGIX Agent</span>
                <span class="message-time">${this.formatTime(timestamp)}</span>
                ${tokenStatsHtml}
            </div>
            <div class="message-content">${this.formatMessage(message)}</div>
        `;
        this.chatMessages.appendChild(messageEl);
        this.scrollToBottom();
    }

    formatTokenCount(count) {
        if (count >= 1000000) {
            return (count / 1000000).toFixed(1) + 'M';
        } else if (count >= 1000) {
            return (count / 1000).toFixed(1) + 'k';
        }
        return count.toString();
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

            // Also update context window indicator
            await this.updateContextWindow();

            // Load memory explorer
            await this.refreshMemory();
        } catch (error) {
            console.error('Failed to load session info:', error);
        }
    }

    async updateContextWindow() {
        try {
            const response = await fetch(`/api/sessions/${encodeURIComponent(this.sessionId)}/context-window`);
            if (!response.ok) {
                console.log(`[Context] Failed to fetch: ${response.status} for session ${this.sessionId}`);
                return;
            }

            const data = await response.json();

            // Update model info (VRAM, quantization, size) - v0.32.1
            await this.updateModelInfo(data.model);

            // Update text display
            const usageText = document.getElementById('contextUsageText');
            const bar = document.getElementById('contextBar');
            const indicator = document.getElementById('contextWindowIndicator');

            if (!usageText || !bar || !indicator) return;

            // Format numbers
            const usedStr = this.formatTokenCount(data.tokens_used);
            const limitStr = this.formatTokenCount(data.context_limit);
            usageText.textContent = `${usedStr} / ${limitStr}`;

            // Update progress bar
            bar.style.width = `${Math.min(100, data.usage_percent)}%`;

            // Update classes based on status
            bar.classList.remove('warning', 'critical');
            indicator.classList.remove('warning', 'critical');

            if (data.is_critical) {
                bar.classList.add('critical');
                indicator.classList.add('critical');
            } else if (data.is_warning) {
                bar.classList.add('warning');
                indicator.classList.add('warning');
            }

            // Show/hide compact button based on warning state
            const compactBtn = document.getElementById('compactBtn');
            if (compactBtn) {
                if (data.is_warning || data.is_critical) {
                    compactBtn.classList.remove('hidden');
                } else {
                    compactBtn.classList.add('hidden');
                }
            }

            // Auto-compact at critical level (95%+) if not already compacted
            if (data.is_critical && !this._autoCompactTriggered) {
                this._autoCompactTriggered = true;
                this.addSystemMessage('⚠️ Context usage critical. Auto-compacting memory...', 'warning');
                await this.compactMemory(true);  // Auto mode
            }
        } catch (error) {
            console.error('Failed to update context window:', error);
        }
    }

    // v0.32.1: Update model info display (VRAM, quantization, size)
    async updateModelInfo(modelName) {
        if (!modelName) return;

        // Cache key to avoid redundant API calls
        // BUT: if cache shows estimated VRAM (vram_gb=0), re-fetch to check if model is now loaded
        const cacheValid = this._lastModelInfoName === modelName &&
                          this._modelInfoCache &&
                          this._modelInfoCache.vram_gb > 0;  // Only use cache if we have actual VRAM

        if (cacheValid) {
            this._displayModelInfo(this._modelInfoCache);
            return;
        }

        try {
            // Force refresh in two cases:
            // 1. Cache was cleared (null) - likely after agent response to get actual VRAM
            // 2. Previous cache had estimated VRAM (vram_gb === 0)
            const cacheWasCleared = !this._modelInfoCache;
            const hadEstimatedVram = this._modelInfoCache &&
                                     this._modelInfoCache.vram_gb !== undefined &&
                                     this._modelInfoCache.vram_gb === 0;
            const forceRefresh = cacheWasCleared || hadEstimatedVram;
            const url = `/api/ollama/model/${encodeURIComponent(modelName)}${forceRefresh ? '?refresh=true' : ''}`;
            const response = await fetch(url);
            if (!response.ok) return;

            const data = await response.json();
            if (!data.available || !data.model) return;

            // Cache the result
            this._lastModelInfoName = modelName;
            this._modelInfoCache = data.model;

            this._displayModelInfo(data.model);
        } catch (error) {
            console.error('Failed to fetch model info:', error);
        }
    }

    _displayModelInfo(info) {
        const quantEl = document.getElementById('modelQuantization');
        const vramEl = document.getElementById('modelVram');
        const sizeEl = document.getElementById('modelParamSize');

        if (quantEl) {
            quantEl.textContent = info.quantization || '-';
            quantEl.title = `Quantization: ${info.quantization || 'Unknown'}`;
        }

        if (vramEl) {
            if (info.vram_gb > 0) {
                // Model is loaded - show actual VRAM
                vramEl.textContent = `${info.vram_gb.toFixed(1)}G`;
                vramEl.title = `VRAM usage: ${info.vram_gb.toFixed(2)} GB (loaded)`;
            } else if (info.size_gb > 0) {
                // Model not loaded - estimate from disk size (close approximation for quantized)
                vramEl.textContent = `~${info.size_gb.toFixed(1)}G`;
                vramEl.title = `VRAM estimate: ~${info.size_gb.toFixed(1)} GB (model not loaded yet)`;
            } else {
                vramEl.textContent = '-';
                vramEl.title = 'VRAM: Not available';
            }
        }

        if (sizeEl) {
            sizeEl.textContent = info.parameter_size || '-';
            sizeEl.title = `Parameters: ${info.parameter_size || 'Unknown'}`;
        }
    }

    async compactMemory(isAuto = false) {
        const compactBtn = document.getElementById('compactBtn');

        try {
            // Update button state
            if (compactBtn) {
                compactBtn.disabled = true;
                compactBtn.classList.add('compacting');
                compactBtn.textContent = '🗜️ Compacting...';
            }

            const response = await fetch(`/api/sessions/${encodeURIComponent(this.sessionId)}/compact`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Compaction failed');
            }

            const result = await response.json();

            if (result.compacted) {
                const savedStr = this.formatTokenCount(result.tokens_saved_estimate);
                this.addSystemMessage(
                    `✅ Memory compacted: ${result.messages_compacted} messages summarized, ~${savedStr} tokens freed.`,
                    'success'
                );

                // Refresh context window display
                await this.updateContextWindow();

                // Reset auto-compact trigger for next cycle
                this._autoCompactTriggered = false;
            } else {
                if (!isAuto) {
                    this.addSystemMessage(`ℹ️ ${result.reason}`, 'info');
                }
            }
        } catch (error) {
            console.error('Failed to compact memory:', error);
            if (!isAuto) {
                this.addSystemMessage(`❌ Compaction failed: ${error.message}`, 'error');
            }
        } finally {
            // Reset button state
            if (compactBtn) {
                compactBtn.disabled = false;
                compactBtn.classList.remove('compacting');
                compactBtn.textContent = '🗜️ Compact';
            }
        }
    }

    // =========================================================================
    // Memory Explorer
    // =========================================================================

    async refreshMemory() {
        try {
            const response = await fetch(`/api/sessions/${encodeURIComponent(this.sessionId)}/episodic`);
            if (!response.ok) return;

            const data = await response.json();

            // Update stats
            const stats = data.stats || {};
            const episodeCount = document.getElementById('memoryEpisodeCount');
            const filesCount = document.getElementById('memoryFilesCount');
            const commandsCount = document.getElementById('memoryCommandsCount');

            if (episodeCount) episodeCount.textContent = stats.total_entries || 0;
            if (filesCount) filesCount.textContent = stats.total_files_touched || 0;
            if (commandsCount) commandsCount.textContent = stats.total_commands_run || 0;

            // Update list
            this.renderMemoryList(data.entries || []);
        } catch (error) {
            console.error('Failed to refresh memory:', error);
        }
    }

    renderMemoryList(entries) {
        const container = document.getElementById('memoryList');
        if (!container) return;

        if (entries.length === 0) {
            container.innerHTML = '<p style="color: var(--text-secondary); font-size: 12px;">No memories yet. Send a message to start building memory.</p>';
            return;
        }

        container.innerHTML = entries.map(entry => {
            const time = new Date(entry.timestamp).toLocaleString('en-US', {
                month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
            });
            const goal = this.escapeHtml(entry.user_goal || 'Unknown goal').substring(0, 60);
            const filesCount = (entry.files_touched || []).length;

            return `
                <div class="memory-entry" onclick="app.showMemoryEntry('${entry.task_id}')" title="${this.escapeHtml(entry.user_goal)}">
                    <button class="memory-entry-delete" onclick="event.stopPropagation(); app.deleteMemoryEntry('${entry.task_id}')" title="Delete">✕</button>
                    <div class="memory-entry-goal">${goal}</div>
                    <div class="memory-entry-meta">
                        <span class="memory-entry-time">${time}</span>
                        <span class="memory-entry-files">${filesCount} files</span>
                    </div>
                </div>
            `;
        }).join('');
    }

    async searchMemory(query) {
        if (!query || query.length < 2) {
            await this.refreshMemory();
            return;
        }

        try {
            const response = await fetch(`/api/sessions/${encodeURIComponent(this.sessionId)}/episodic/search?q=${encodeURIComponent(query)}`);
            if (!response.ok) return;

            const data = await response.json();
            this.renderMemoryList(data.results || []);
        } catch (error) {
            console.error('Failed to search memory:', error);
        }
    }

    async showMemoryEntry(taskId) {
        try {
            const response = await fetch(`/api/sessions/${encodeURIComponent(this.sessionId)}/episodic/${encodeURIComponent(taskId)}`);
            if (!response.ok) return;

            const data = await response.json();
            const entry = data.entry;

            // Format entry details
            const details = `
## Episode: ${entry.user_goal}

**Task ID:** ${entry.task_id}
**Time:** ${new Date(entry.timestamp).toLocaleString()}

### Plan
${entry.plan_summary || 'No plan recorded'}

### Result
${entry.result_summary || 'No result recorded'}

### Key Decisions
${(entry.key_decisions || []).map(d => `- ${d}`).join('\n') || 'None'}

### Files Touched
${(entry.files_touched || []).map(f => `- \`${f}\``).join('\n') || 'None'}

### Commands Run
${(entry.commands_run || []).slice(0, 10).map(c => `- \`${c.substring(0, 80)}\``).join('\n') || 'None'}

### Open Questions
${(entry.open_questions || []).map(q => `- ${q}`).join('\n') || 'None'}
            `.trim();

            // Show in a system message or modal
            this.addSystemMessage('📚 Memory Entry Details:\n\n' + details, 'info');
        } catch (error) {
            console.error('Failed to show memory entry:', error);
        }
    }

    async deleteMemoryEntry(taskId) {
        if (!confirm('Delete this memory entry?')) return;

        try {
            const response = await fetch(`/api/sessions/${encodeURIComponent(this.sessionId)}/episodic/${encodeURIComponent(taskId)}`, {
                method: 'DELETE'
            });

            if (response.ok) {
                this.addSystemMessage('🗑️ Memory entry deleted', 'success');
                await this.refreshMemory();
            }
        } catch (error) {
            console.error('Failed to delete memory entry:', error);
        }
    }

    async clearMemory() {
        if (!confirm('Clear ALL memory entries? This cannot be undone.')) return;

        try {
            const response = await fetch(`/api/sessions/${encodeURIComponent(this.sessionId)}/episodic`, {
                method: 'DELETE'
            });

            if (response.ok) {
                const data = await response.json();
                this.addSystemMessage(`🗑️ Cleared ${data.deleted} memory entries`, 'success');
                await this.refreshMemory();
            }
        } catch (error) {
            console.error('Failed to clear memory:', error);
        }
    }

    setupMemorySearch() {
        const searchInput = document.getElementById('memorySearchInput');
        if (!searchInput) return;

        let debounceTimer;
        searchInput.addEventListener('input', (e) => {
            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(() => {
                this.searchMemory(e.target.value.trim());
            }, 300);
        });
    }

    // =========================================================================
    // File Handling
    // =========================================================================

    setupFileHandling() {
        const dropZone = document.getElementById('fileDropZone');
        const fileInput = document.getElementById('fileInput');
        const chatArea = document.querySelector('.chat-area') || document.querySelector('main');

        if (!dropZone || !fileInput || !chatArea) return;

        // Show drop zone when dragging files over chat area
        chatArea.addEventListener('dragenter', (e) => {
            e.preventDefault();
            dropZone.classList.remove('hidden');
            dropZone.classList.add('active');
        });

        chatArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('active');
        });

        chatArea.addEventListener('dragleave', (e) => {
            // Only hide if leaving the chat area entirely
            if (!chatArea.contains(e.relatedTarget)) {
                dropZone.classList.remove('active');
                if (this.pendingFiles.length === 0) {
                    dropZone.classList.add('hidden');
                }
            }
        });

        // Handle drop
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('active');
            this.handleFiles(e.dataTransfer.files);
        });

        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
        });

        // Handle file input selection
        fileInput.addEventListener('change', (e) => {
            this.handleFiles(e.target.files);
            fileInput.value = ''; // Reset for next selection
        });
    }

    async handleFiles(files) {
        const dropZone = document.getElementById('fileDropZone');
        const preview = document.getElementById('filePreview');

        for (const file of files) {
            // Check if already added
            if (this.pendingFiles.find(f => f.name === file.name && f.size === file.size)) {
                continue;
            }

            // Check file type
            const fileInfo = this.getFileInfo(file);
            if (!fileInfo.supported) {
                this.addSystemMessage(`Unsupported file type: ${file.name}`, 'warning');
                continue;
            }

            // v0.33: Handle archives separately - offer to index to RAG
            if (fileInfo.isArchive) {
                this.handleArchiveFile(file, fileInfo);
                continue;
            }

            // Add to pending files
            const fileData = {
                file: file,
                name: file.name,
                size: file.size,
                type: fileInfo.type,
                icon: fileInfo.icon,
                content: null,
                error: null,
                converting: fileInfo.needsConversion
            };

            this.pendingFiles.push(fileData);
            this.updateFilePreview();

            // Read file content
            try {
                if (fileInfo.needsConversion) {
                    // Send to server for conversion
                    fileData.content = await this.convertFile(file);
                } else {
                    // Read as text directly
                    fileData.content = await this.readFileAsText(file);
                }
                fileData.converting = false;
            } catch (error) {
                fileData.error = error.message;
                fileData.converting = false;
            }

            this.updateFilePreview();
        }

        // Show preview if files are pending
        if (this.pendingFiles.length > 0) {
            dropZone.classList.add('hidden');
            preview.classList.remove('hidden');
        }
    }

    /**
     * v0.33: Handle archive files (ZIP, TAR) - upload to RAG indexer
     */
    async handleArchiveFile(file, fileInfo) {
        const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
        this.addSystemMessage(`📦 Archive detected: ${file.name} (${sizeMB} MB). Uploading to RAG index...`, 'info');

        // Get chunk parameters from UI
        const chunkSize = parseInt(document.getElementById('ragChunkSize')?.value || '1000', 10);
        const chunkOverlap = parseInt(document.getElementById('ragChunkOverlap')?.value || '200', 10);

        // Get converter settings from UI
        const pdfEnabled = document.getElementById('ragPdfEnabled')?.checked ?? true;
        const pandocEnabled = document.getElementById('ragPandocEnabled')?.checked ?? true;

        try {
            const formData = new FormData();
            formData.append('file', file);
            formData.append('session_id', this.sessionId);
            formData.append('chunk_size', chunkSize.toString());
            formData.append('chunk_overlap', chunkOverlap.toString());
            formData.append('pdf_enabled', pdfEnabled.toString());
            formData.append('pandoc_enabled', pandocEnabled.toString());

            const response = await fetch('/api/rag/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Upload failed');
            }

            const result = await response.json();
            const converted = result.files_converted || 0;
            const convertedInfo = converted > 0 ? ` | Converted: ${converted}` : '';
            this.addSystemMessage(
                `✅ Indexed ${result.files_indexed || 0} files from ${file.name}\n` +
                `   Chunks: ${result.chunks_created || 0} | Skipped: ${result.files_skipped || 0}${convertedInfo}`,
                'success'
            );

            // Refresh RAG status
            await this.loadRagStatus();

        } catch (error) {
            this.addSystemMessage(`❌ Failed to index ${file.name}: ${error.message}`, 'error');
        }
    }

    getFileInfo(file) {
        const ext = file.name.split('.').pop().toLowerCase();

        // v0.33: Extended file type support including code files and archives
        const textExtensions = [
            // General text
            'txt', 'md', 'rst', 'log', 'csv',
            // Config files
            'json', 'yaml', 'yml', 'xml', 'toml', 'ini', 'cfg', 'conf', 'env', 'properties',
            // Web
            'html', 'htm', 'css', 'scss', 'less',
            // JavaScript/TypeScript
            'js', 'ts', 'jsx', 'tsx', 'mjs', 'cjs', 'vue', 'svelte',
            // Python
            'py', 'pyw', 'pyx', 'pxd', 'pyi',
            // Java/JVM
            'java', 'kt', 'kts', 'groovy', 'gradle', 'scala',
            // C/C++
            'c', 'h', 'cpp', 'hpp', 'cc', 'hh', 'cxx', 'hxx',
            // Shell/Scripts
            'sh', 'bash', 'zsh', 'fish', 'ps1', 'bat', 'cmd',
            // MATLAB/Octave
            'm', 'mat',
            // Build/Project files
            'pom', 'makefile', 'cmake', 'dockerfile', 'vagrantfile',
            // SQL
            'sql', 'ddl', 'dml',
            // Other languages
            'rb', 'php', 'go', 'rs', 'swift', 'r', 'jl', 'lua', 'pl', 'pm',
            // Git
            'gitignore', 'gitattributes', 'gitmodules',
        ];

        const conversionExtensions = ['pdf', 'docx', 'doc', 'odt', 'rtf'];

        // v0.33: Archive extensions for RAG indexing
        const archiveExtensions = ['zip', 'tar', 'gz', 'tgz'];

        const icons = {
            'py': '🐍', 'js': '📜', 'ts': '📘', 'json': '📋', 'yaml': '📄', 'yml': '📄',
            'md': '📝', 'txt': '📄', 'html': '🌐', 'css': '🎨', 'sql': '🗃️',
            'pdf': '📕', 'docx': '📘', 'doc': '📘', 'odt': '📗',
            'sh': '🖥️', 'bash': '🖥️', 'csv': '📊',
            'java': '☕', 'kt': '🟣', 'xml': '📰', 'pom': '📦',
            'm': '📐', 'c': '⚙️', 'cpp': '⚙️', 'h': '⚙️',
            'go': '🐹', 'rs': '🦀', 'rb': '💎', 'php': '🐘',
            'zip': '📦', 'tar': '📦', 'gz': '📦', 'tgz': '📦',
        };

        if (textExtensions.includes(ext)) {
            return { supported: true, type: 'text', icon: icons[ext] || '📄', needsConversion: false, isArchive: false };
        } else if (conversionExtensions.includes(ext)) {
            return { supported: true, type: ext, icon: icons[ext] || '📄', needsConversion: true, isArchive: false };
        } else if (archiveExtensions.includes(ext)) {
            return { supported: true, type: 'archive', icon: icons[ext] || '📦', needsConversion: false, isArchive: true };
        }

        return { supported: false, type: 'unknown', icon: '❓', needsConversion: false, isArchive: false };
    }

    readFileAsText(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => resolve(e.target.result);
            reader.onerror = () => reject(new Error('Failed to read file'));
            reader.readAsText(file);
        });
    }

    async convertFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('/api/files/convert', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Conversion failed');
        }

        const data = await response.json();
        return data.content;
    }

    updateFilePreview() {
        const preview = document.getElementById('filePreview');
        if (!preview) return;

        if (this.pendingFiles.length === 0) {
            preview.classList.add('hidden');
            preview.innerHTML = '';
            return;
        }

        preview.innerHTML = this.pendingFiles.map((f, idx) => `
            <div class="file-preview-item ${f.converting ? 'converting' : ''} ${f.error ? 'error' : ''}">
                <span class="file-icon">${f.icon}</span>
                <span class="file-name" title="${f.name}">${f.name}</span>
                <span class="file-size">${this.formatFileSize(f.size)}</span>
                ${f.converting ? '<span class="file-status">⏳</span>' : ''}
                ${f.error ? '<span class="file-status" title="' + f.error + '">⚠️</span>' : ''}
                <span class="file-remove" onclick="app.removeFile(${idx})">✕</span>
            </div>
        `).join('');

        preview.classList.remove('hidden');
    }

    formatFileSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    }

    removeFile(index) {
        this.pendingFiles.splice(index, 1);
        this.updateFilePreview();

        if (this.pendingFiles.length === 0) {
            const dropZone = document.getElementById('fileDropZone');
            if (dropZone) dropZone.classList.add('hidden');
        }
    }

    getFileContextForMessage() {
        // Get file contents to prepend to message
        const validFiles = this.pendingFiles.filter(f => f.content && !f.error);
        if (validFiles.length === 0) return '';

        const fileContext = validFiles.map(f => {
            const preview = f.content.length > 10000
                ? f.content.substring(0, 10000) + '\n\n[... truncated, ' + (f.content.length - 10000) + ' more characters ...]'
                : f.content;
            return `--- File: ${f.name} ---\n${preview}\n--- End of ${f.name} ---`;
        }).join('\n\n');

        // Clear pending files after including in message
        this.pendingFiles = [];
        this.updateFilePreview();

        return fileContext;
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
        // Prevent double-clicks / concurrent calls
        if (this._savingSettings) {
            console.log('saveSettings: already in progress, ignoring');
            return;
        }
        this._savingSettings = true;

        const sessionId = document.getElementById('sessionIdInput').value || this.sessionId;
        const sandbox = document.getElementById('sandboxInput').value;
        const model = document.getElementById('modelInput').value;
        const profile = document.getElementById('profileInput').value;

        try {
            // Update existing session (PUT) or create new one (POST)
            const isNewSession = sessionId !== this.sessionId;
            const method = isNewSession ? 'POST' : 'PUT';
            const url = isNewSession ? '/api/sessions' : `/api/sessions/${encodeURIComponent(this.sessionId)}`;

            console.log(`saveSettings: ${method} ${url} (isNew=${isNewSession}, input=${sessionId}, current=${this.sessionId})`);

            const response = await fetch(url, {
                method: method,
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    sandbox_root: sandbox,
                    model: model,
                    profile: profile
                })
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`HTTP ${response.status}: ${errorText}`);
            }

            const data = await response.json();

            // Clear model info cache so VRAM updates
            this._lastModelInfoName = null;
            this._modelInfoCache = null;

            if (isNewSession) {
                this.sessionId = data.session_id;
                // Reconnect only if new session
                if (this.ws) {
                    this.ws.close();
                }
                this.connect();
            }

            this.loadSessionInfo();
            this.closeSettings();

            this.addSystemMessage(`Settings ${isNewSession ? 'saved' : 'updated'}. Model: ${model}`);
        } catch (error) {
            console.error('Failed to save settings:', error);
            this.addSystemMessage(`Failed to update settings: ${error.message}`, 'error');
        } finally {
            this._savingSettings = false;
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

    // ============================
    // Thread Management (v0.33)
    // ============================

    /**
     * Load and display threads in the sidebar
     */
    async loadThreads() {
        const threadList = document.getElementById('threadList');
        if (!threadList) return;

        try {
            const resp = await fetch(`/api/sessions/${encodeURIComponent(this.sessionId)}/threads`);
            if (!resp.ok) {
                throw new Error(`HTTP ${resp.status}`);
            }

            const data = await resp.json();
            const threads = data.threads || [];

            if (threads.length === 0) {
                threadList.innerHTML = `
                    <div class="thread-empty">
                        No threads yet.<br>Click "New Thread" to start.
                    </div>
                `;
                return;
            }

            threadList.innerHTML = threads.map(t => `
                <div class="thread-item ${t.is_active ? 'active' : ''}"
                     onclick="app.switchThread('${t.id}')"
                     data-thread-id="${t.id}">
                    <div class="thread-item-header">
                        <span class="thread-item-name" title="${this.escapeHtml(t.name)}">${this.escapeHtml(t.name)}</span>
                        <span class="thread-item-count">${t.message_count}</span>
                    </div>
                    <div class="thread-item-meta">
                        <span class="thread-item-time">${this.formatThreadTime(t.updated_at)}</span>
                    </div>
                    <div class="thread-item-actions">
                        <button class="thread-item-btn export" onclick="event.stopPropagation(); app.exportThread('${t.id}')" title="Export">📥</button>
                        <button class="thread-item-btn" onclick="event.stopPropagation(); app.deleteThread('${t.id}')" title="Delete">🗑</button>
                    </div>
                </div>
            `).join('');
        } catch (error) {
            console.error('Failed to load threads:', error);
            threadList.innerHTML = `
                <p style="color: var(--text-secondary); font-size: 12px;">
                    Failed to load threads
                </p>
            `;
        }
    }

    /**
     * Format timestamp for thread display
     */
    formatThreadTime(isoString) {
        if (!isoString) return '';
        try {
            const date = new Date(isoString);
            const now = new Date();
            const diff = now - date;

            // Less than 1 hour ago
            if (diff < 3600000) {
                const mins = Math.floor(diff / 60000);
                return mins <= 1 ? 'Just now' : `${mins}m ago`;
            }
            // Less than 24 hours
            if (diff < 86400000) {
                const hours = Math.floor(diff / 3600000);
                return `${hours}h ago`;
            }
            // Same year
            if (date.getFullYear() === now.getFullYear()) {
                return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
            }
            // Different year
            return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: '2-digit' });
        } catch (e) {
            return '';
        }
    }

    /**
     * Create a new thread
     */
    async createThread() {
        const name = prompt('Enter thread name (optional):');
        // If user cancels, don't create
        if (name === null) return;

        try {
            const resp = await fetch(`/api/sessions/${encodeURIComponent(this.sessionId)}/threads`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: name || null })
            });

            if (!resp.ok) {
                throw new Error(`HTTP ${resp.status}`);
            }

            const data = await resp.json();
            console.log('Created thread:', data);

            // Clear chat and reload threads
            this.clearChat();
            await this.loadThreads();

            this.addSystemMessage(`Created new thread: ${data.name}`);
        } catch (error) {
            console.error('Failed to create thread:', error);
            this.addSystemMessage('Failed to create thread', 'error');
        }
    }

    /**
     * Switch to a different thread
     */
    async switchThread(threadId) {
        try {
            const resp = await fetch(`/api/sessions/${encodeURIComponent(this.sessionId)}/threads/active/${threadId}`, {
                method: 'PUT'
            });

            if (!resp.ok) {
                throw new Error(`HTTP ${resp.status}`);
            }

            const data = await resp.json();
            console.log('Switched to thread:', data);

            // Clear current chat and load thread messages
            this.clearChat();
            await this.loadThreadMessages(threadId);
            await this.loadThreads();  // Refresh to show active state

        } catch (error) {
            console.error('Failed to switch thread:', error);
            this.addSystemMessage('Failed to switch thread', 'error');
        }
    }

    /**
     * Load messages from a thread into the chat
     */
    async loadThreadMessages(threadId) {
        try {
            const resp = await fetch(`/api/sessions/${encodeURIComponent(this.sessionId)}/threads/${threadId}/messages?limit=100`);
            if (!resp.ok) {
                throw new Error(`HTTP ${resp.status}`);
            }

            const data = await resp.json();
            const messages = data.messages || [];

            // Display messages in chat
            for (const msg of messages) {
                if (msg.role === 'user') {
                    this.addUserMessage(msg.content, msg.timestamp);
                } else if (msg.role === 'assistant') {
                    this.addAgentMessage(msg.content, msg.timestamp);
                } else if (msg.role === 'system') {
                    this.addSystemMessage(msg.content);
                }
            }

            if (messages.length > 0) {
                this.scrollToBottom();
            }
        } catch (error) {
            console.error('Failed to load thread messages:', error);
        }
    }

    /**
     * Delete a thread
     */
    async deleteThread(threadId) {
        if (!confirm('Delete this thread? This cannot be undone.')) return;

        try {
            const resp = await fetch(`/api/sessions/${encodeURIComponent(this.sessionId)}/threads/${threadId}`, {
                method: 'DELETE'
            });

            if (!resp.ok) {
                throw new Error(`HTTP ${resp.status}`);
            }

            console.log('Deleted thread:', threadId);
            await this.loadThreads();
            this.addSystemMessage('Thread deleted');
        } catch (error) {
            console.error('Failed to delete thread:', error);
            this.addSystemMessage('Failed to delete thread', 'error');
        }
    }

    /**
     * Export a thread as markdown
     */
    async exportThread(threadId) {
        try {
            const resp = await fetch(`/api/sessions/${encodeURIComponent(this.sessionId)}/threads/${threadId}/export?format=markdown`);
            if (!resp.ok) {
                throw new Error(`HTTP ${resp.status}`);
            }

            const text = await resp.text();

            // Create download link
            const blob = new Blob([text], { type: 'text/markdown' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `thread-${threadId}.md`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);

            this.addSystemMessage('Thread exported to markdown');
        } catch (error) {
            console.error('Failed to export thread:', error);
            this.addSystemMessage('Failed to export thread', 'error');
        }
    }

    /**
     * Helper to escape HTML entities
     */
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // ============================
    // RAG Management (v0.33)
    // ============================

    /**
     * Load and display RAG status
     */
    async loadRagStatus() {
        try {
            const resp = await fetch(`/api/rag/status?session_id=${encodeURIComponent(this.sessionId)}`);
            if (!resp.ok) {
                throw new Error(`HTTP ${resp.status}`);
            }

            const data = await resp.json();

            // Update toggle
            const toggle = document.getElementById('ragEnabledToggle');
            if (toggle) {
                toggle.checked = data.enabled;
            }

            // Update stats
            const indexStatus = document.getElementById('ragIndexStatus');
            if (indexStatus) {
                if (data.index_exists) {
                    indexStatus.textContent = data.index_loaded ? 'Loaded' : 'Ready';
                    indexStatus.className = 'stat-value ' + (data.index_loaded ? 'loaded' : '');
                } else {
                    indexStatus.textContent = 'None';
                    indexStatus.className = 'stat-value not-loaded';
                }
            }

            const docCount = document.getElementById('ragDocCount');
            if (docCount) {
                docCount.textContent = data.document_count || '0';
            }

            const chunkCount = document.getElementById('ragChunkCount');
            if (chunkCount) {
                chunkCount.textContent = data.chunk_count || '0';
            }

        } catch (error) {
            console.error('Failed to load RAG status:', error);
        }
    }

    /**
     * Toggle RAG enabled state
     */
    async toggleRag(enabled) {
        try {
            const resp = await fetch(`/api/rag/enable?session_id=${encodeURIComponent(this.sessionId)}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ enabled })
            });

            if (!resp.ok) {
                throw new Error(`HTTP ${resp.status}`);
            }

            const data = await resp.json();
            console.log('RAG toggle:', data);

            // Refresh status
            await this.loadRagStatus();

            this.addSystemMessage(`RAG ${enabled ? 'enabled' : 'disabled'}`);
        } catch (error) {
            console.error('Failed to toggle RAG:', error);
            this.addSystemMessage('Failed to toggle RAG', 'error');

            // Revert toggle
            const toggle = document.getElementById('ragEnabledToggle');
            if (toggle) {
                toggle.checked = !enabled;
            }
        }
    }

    /**
     * v0.33: Index chat history to RAG
     */
    async indexChatToRag() {
        const chunkSize = parseInt(document.getElementById('ragChunkSize')?.value || '1000', 10);

        try {
            this.addSystemMessage('💬→📚 Indexing chat history to RAG...', 'info');

            const response = await fetch(`/api/rag/index-chat?session_id=${encodeURIComponent(this.sessionId)}&chunk_size=${chunkSize}`, {
                method: 'POST'
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Indexing failed');
            }

            const result = await response.json();
            this.addSystemMessage(
                `✅ Chat indexed: ${result.chunks_created || 0} chunks from ${result.messages_processed || 0} messages`,
                'success'
            );

            // Refresh RAG status
            await this.loadRagStatus();

        } catch (error) {
            this.addSystemMessage(`❌ Failed to index chat: ${error.message}`, 'error');
        }
    }

    /**
     * v0.33: Clear RAG index
     */
    async clearRagIndex() {
        if (!confirm('Clear the entire RAG index? This cannot be undone.')) {
            return;
        }

        try {
            const response = await fetch(`/api/rag/index?session_id=${encodeURIComponent(this.sessionId)}`, {
                method: 'DELETE'
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Clear failed');
            }

            this.addSystemMessage('🗑️ RAG index cleared', 'success');

            // Refresh RAG status
            await this.loadRagStatus();

        } catch (error) {
            this.addSystemMessage(`❌ Failed to clear index: ${error.message}`, 'error');
        }
    }

    /**
     * v0.33: Handle RAG file upload from file input
     */
    async handleRagFileUpload(event) {
        const files = event.target.files;
        if (!files || files.length === 0) return;

        // Get chunk parameters from UI
        const chunkSize = parseInt(document.getElementById('ragChunkSize')?.value || '1000', 10);
        const chunkOverlap = parseInt(document.getElementById('ragChunkOverlap')?.value || '200', 10);
        const pdfEnabled = document.getElementById('ragPdfEnabled')?.checked ?? true;
        const pandocEnabled = document.getElementById('ragPandocEnabled')?.checked ?? true;

        let totalIndexed = 0;
        let totalChunks = 0;
        let totalConverted = 0;
        let totalSkipped = 0;

        this.addSystemMessage(`📤 Uploading ${files.length} file(s) to RAG index...`, 'info');

        for (const file of files) {
            try {
                const formData = new FormData();
                formData.append('file', file);
                formData.append('session_id', this.sessionId);
                formData.append('chunk_size', chunkSize.toString());
                formData.append('chunk_overlap', chunkOverlap.toString());
                formData.append('pdf_enabled', pdfEnabled.toString());
                formData.append('pandoc_enabled', pandocEnabled.toString());

                const response = await fetch('/api/rag/upload', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const error = await response.json();
                    this.addSystemMessage(`⚠️ Failed to index ${file.name}: ${error.detail}`, 'warning');
                    continue;
                }

                const result = await response.json();
                totalIndexed += result.files_indexed || 0;
                totalChunks += result.chunks_created || 0;
                totalConverted += result.files_converted || 0;
                totalSkipped += result.files_skipped || 0;

            } catch (error) {
                this.addSystemMessage(`⚠️ Error uploading ${file.name}: ${error.message}`, 'warning');
            }
        }

        // Summary message
        const convertedInfo = totalConverted > 0 ? ` | Converted: ${totalConverted}` : '';
        this.addSystemMessage(
            `✅ RAG indexing complete\n` +
            `   Files: ${totalIndexed} | Chunks: ${totalChunks} | Skipped: ${totalSkipped}${convertedInfo}`,
            'success'
        );

        // Refresh RAG status
        await this.loadRagStatus();

        // Clear the file input for re-use
        event.target.value = '';
    }

    /**
     * Show RAG browser (documents and chunks)
     */
    async showRagBrowser() {
        try {
            // First get stats
            const statsResp = await fetch(`/api/rag/stats?session_id=${encodeURIComponent(this.sessionId)}`);
            const stats = await statsResp.json();

            // Get documents list
            const docsResp = await fetch(`/api/rag/documents?session_id=${encodeURIComponent(this.sessionId)}&limit=20`);
            const docs = await docsResp.json();

            // Build display message
            let message = '## RAG Index Browser\n\n';

            if (!stats.exists) {
                message += '**No index found.**\n\n';
                message += 'Create an index by adding documents or running `ragix-ast scan`.';
            } else {
                message += `**Index Path:** \`${stats.index_path}\`\n\n`;
                message += `**Total Size:** ${stats.total_size_kb || 0} KB\n\n`;

                if (stats.files && Object.keys(stats.files).length > 0) {
                    message += '### Index Files\n\n';
                    for (const [name, info] of Object.entries(stats.files)) {
                        message += `- \`${name}\`: ${info.size_kb} KB\n`;
                    }
                }

                if (docs.documents && docs.documents.length > 0) {
                    message += '\n### Indexed Documents\n\n';
                    message += `Showing ${docs.documents.length} of ${docs.total} documents.\n\n`;
                    for (const doc of docs.documents.slice(0, 10)) {
                        message += `- ${doc.path || doc.file_path || doc.name || 'Unknown'}\n`;
                    }
                    if (docs.total > 10) {
                        message += `\n... and ${docs.total - 10} more`;
                    }
                } else {
                    message += '\n### No Documents Indexed\n\n';
                    message += 'Use `ragix-ast scan <path>` to index files.';
                }
            }

            // Display in chat
            this.addAgentMessage(message);

        } catch (error) {
            console.error('Failed to show RAG browser:', error);
            this.addSystemMessage('Failed to load RAG index', 'error');
        }
    }

    // ==================== v0.33: Global Context Methods ====================

    /**
     * Initialize the global context editor
     */
    initGlobalContext() {
        const textarea = document.getElementById('globalContextInput');
        const charCount = document.getElementById('contextCharCount');

        if (textarea && charCount) {
            // Update char count on input
            textarea.addEventListener('input', () => {
                const len = textarea.value.length;
                charCount.textContent = `${len} chars`;

                // Color coding for length
                charCount.className = 'context-char-count';
                if (len > 1000) {
                    charCount.classList.add('error');
                } else if (len > 500) {
                    charCount.classList.add('warning');
                }
            });

            // Load saved context
            this.loadGlobalContext();
        }
    }

    /**
     * Load global context from server or localStorage
     */
    async loadGlobalContext() {
        const textarea = document.getElementById('globalContextInput');
        const charCount = document.getElementById('contextCharCount');

        if (!textarea) return;

        try {
            // Try to load from server API
            const response = await fetch(`/api/sessions/${encodeURIComponent(this.sessionId)}/context`);
            if (response.ok) {
                const data = await response.json();
                if (data.system_instructions) {
                    textarea.value = data.system_instructions;
                    if (charCount) {
                        charCount.textContent = `${data.system_instructions.length} chars`;
                    }
                    return;
                }
            }
        } catch (error) {
            console.warn('Failed to load context from server:', error);
        }

        // Fallback to localStorage
        const saved = localStorage.getItem('ragix_global_context');
        if (saved) {
            textarea.value = saved;
            if (charCount) {
                charCount.textContent = `${saved.length} chars`;
            }
        }
    }

    /**
     * Save global context to server and localStorage
     */
    async saveGlobalContext() {
        const textarea = document.getElementById('globalContextInput');
        if (!textarea) return;

        const context = textarea.value.trim();

        // Save to localStorage as backup
        localStorage.setItem('ragix_global_context', context);

        try {
            // Save to server
            const response = await fetch(`/api/sessions/${encodeURIComponent(this.sessionId)}/context`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    system_instructions: context
                })
            });

            if (response.ok) {
                this.addSystemMessage('Global context saved', 'success');
            } else {
                throw new Error('Server returned error');
            }
        } catch (error) {
            console.error('Failed to save context to server:', error);
            this.addSystemMessage('Context saved locally (server unavailable)', 'warning');
        }
    }

    /**
     * Get current global context
     */
    getGlobalContext() {
        const textarea = document.getElementById('globalContextInput');
        return textarea ? textarea.value.trim() : '';
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
