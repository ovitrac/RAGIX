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

            case 'agent_message':
                this.addAgentMessage(message, timestamp);
                break;

            case 'tool_call':
                this.addToolTrace(data);
                break;

            case 'error':
                this.addSystemMessage(message, 'error');
                break;

            case 'pong':
                // Heartbeat response
                break;

            default:
                console.warn('Unknown message type:', type);
        }
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
            const response = await fetch(`/api/sessions`);
            const data = await response.json();

            const session = data.sessions.find(s => s.id === this.sessionId);
            if (session) {
                document.getElementById('sandboxPath').textContent = session.sandbox_root;
                document.getElementById('modelName').textContent = session.model;
                document.getElementById('profileName').textContent = session.profile;
            }
        } catch (error) {
            console.error('Failed to load session info:', error);
        }
    }

    openSettings() {
        this.settingsModal.classList.remove('hidden');
    }

    closeSettings() {
        this.settingsModal.classList.add('hidden');
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

        // Code blocks
        text = text.replace(/```(\w+)?\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');

        // Inline code
        text = text.replace(/`([^`]+)`/g, '<code>$1</code>');

        // Bold
        text = text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

        // Italic
        text = text.replace(/\*([^*]+)\*/g, '<em>$1</em>');

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
}

// Initialize app when DOM is ready
const app = new RAGIXApp();

// Heartbeat to keep connection alive
setInterval(() => {
    if (app.ws && app.ws.readyState === WebSocket.OPEN) {
        app.ws.send(JSON.stringify({ type: 'ping' }));
    }
}, 30000);
