/**
 * RAGIX Log Viewer - Real-time log streaming and filtering
 *
 * Provides real-time log viewing with filtering by level, agent,
 * tool, search, and export capabilities.
 *
 * Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-25
 */

class LogViewer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.logs = [];
        this.filteredLogs = [];
        this.filters = {
            level: 'all',
            agent: 'all',
            tool: 'all',
            search: ''
        };
        this.autoScroll = true;
        this.maxLogs = 1000;
        this.ws = null;

        // Log level colors
        this.levelColors = {
            'DEBUG': '#6b7280',
            'INFO': '#3b82f6',
            'WARNING': '#f59e0b',
            'ERROR': '#ef4444',
            'CRITICAL': '#dc2626'
        };

        this.init();
    }

    init() {
        this.container.innerHTML = `
            <div class="log-viewer">
                <div class="log-toolbar">
                    <div class="log-filters">
                        <select id="logLevelFilter" onchange="logViewer.filterByLevel(this.value)">
                            <option value="all">All Levels</option>
                            <option value="DEBUG">DEBUG</option>
                            <option value="INFO">INFO</option>
                            <option value="WARNING">WARNING</option>
                            <option value="ERROR">ERROR</option>
                            <option value="CRITICAL">CRITICAL</option>
                        </select>
                        <select id="logAgentFilter" onchange="logViewer.filterByAgent(this.value)">
                            <option value="all">All Agents</option>
                        </select>
                        <select id="logToolFilter" onchange="logViewer.filterByTool(this.value)">
                            <option value="all">All Tools</option>
                        </select>
                        <input type="text" id="logSearch" placeholder="Search logs..."
                               onkeyup="logViewer.filterBySearch(this.value)">
                    </div>
                    <div class="log-actions">
                        <label class="log-autoscroll">
                            <input type="checkbox" id="logAutoScroll" checked
                                   onchange="logViewer.toggleAutoScroll(this.checked)">
                            Auto-scroll
                        </label>
                        <button class="log-btn" onclick="logViewer.clear()">Clear</button>
                        <button class="log-btn" onclick="logViewer.exportLogs()">Export</button>
                    </div>
                </div>
                <div class="log-content" id="logContent">
                    <div class="log-empty">No logs yet</div>
                </div>
                <div class="log-status">
                    <span id="logCount">0 logs</span>
                    <span id="logStreamStatus" class="log-stream-status">Disconnected</span>
                </div>
            </div>
        `;

        this.logContent = document.getElementById('logContent');
        this.agentFilter = document.getElementById('logAgentFilter');
        this.toolFilter = document.getElementById('logToolFilter');
    }

    /**
     * Add a log entry.
     *
     * @param {Object} logEntry - Log entry object
     * @param {string} logEntry.timestamp - ISO timestamp
     * @param {string} logEntry.level - Log level
     * @param {string} logEntry.message - Log message
     * @param {string} logEntry.agent - Agent name (optional)
     * @param {string} logEntry.tool - Tool name (optional)
     * @param {Object} logEntry.extra - Extra data (optional)
     */
    addLog(logEntry) {
        // Enforce max logs
        if (this.logs.length >= this.maxLogs) {
            this.logs.shift();
        }

        this.logs.push(logEntry);

        // Update filter options
        this.updateFilterOptions(logEntry);

        // Apply filters and render if matches
        if (this.matchesFilters(logEntry)) {
            this.filteredLogs.push(logEntry);
            this.renderLogEntry(logEntry);

            if (this.autoScroll) {
                this.scrollToBottom();
            }
        }

        this.updateCount();
    }

    /**
     * Add multiple logs at once.
     */
    addLogs(logEntries) {
        logEntries.forEach(entry => this.addLog(entry));
    }

    updateFilterOptions(logEntry) {
        // Update agent filter
        if (logEntry.agent && !this.hasOption(this.agentFilter, logEntry.agent)) {
            const option = document.createElement('option');
            option.value = logEntry.agent;
            option.textContent = logEntry.agent;
            this.agentFilter.appendChild(option);
        }

        // Update tool filter
        if (logEntry.tool && !this.hasOption(this.toolFilter, logEntry.tool)) {
            const option = document.createElement('option');
            option.value = logEntry.tool;
            option.textContent = logEntry.tool;
            this.toolFilter.appendChild(option);
        }
    }

    hasOption(select, value) {
        return Array.from(select.options).some(opt => opt.value === value);
    }

    matchesFilters(logEntry) {
        // Level filter
        if (this.filters.level !== 'all' && logEntry.level !== this.filters.level) {
            return false;
        }

        // Agent filter
        if (this.filters.agent !== 'all' && logEntry.agent !== this.filters.agent) {
            return false;
        }

        // Tool filter
        if (this.filters.tool !== 'all' && logEntry.tool !== this.filters.tool) {
            return false;
        }

        // Search filter
        if (this.filters.search) {
            const searchLower = this.filters.search.toLowerCase();
            const searchableText = `${logEntry.message} ${logEntry.agent || ''} ${logEntry.tool || ''}`.toLowerCase();
            if (!searchableText.includes(searchLower)) {
                return false;
            }
        }

        return true;
    }

    renderLogEntry(logEntry) {
        // Remove empty message if present
        const empty = this.logContent.querySelector('.log-empty');
        if (empty) empty.remove();

        const entry = document.createElement('div');
        entry.className = `log-entry log-level-${logEntry.level.toLowerCase()}`;

        const time = this.formatTime(logEntry.timestamp);
        const levelColor = this.levelColors[logEntry.level] || '#6b7280';

        let metaHtml = '';
        if (logEntry.agent) {
            metaHtml += `<span class="log-agent">[${logEntry.agent}]</span>`;
        }
        if (logEntry.tool) {
            metaHtml += `<span class="log-tool">${logEntry.tool}</span>`;
        }

        entry.innerHTML = `
            <span class="log-time">${time}</span>
            <span class="log-level" style="color: ${levelColor}">${logEntry.level}</span>
            ${metaHtml}
            <span class="log-message">${this.escapeHtml(logEntry.message)}</span>
        `;

        // Add click handler for extra data
        if (logEntry.extra) {
            entry.classList.add('log-expandable');
            entry.addEventListener('click', () => this.showLogDetails(logEntry));
        }

        this.logContent.appendChild(entry);
    }

    showLogDetails(logEntry) {
        // Create modal for log details
        const modal = document.createElement('div');
        modal.className = 'log-modal';
        modal.innerHTML = `
            <div class="log-modal-content">
                <div class="log-modal-header">
                    <span>Log Details</span>
                    <button onclick="this.closest('.log-modal').remove()">&times;</button>
                </div>
                <div class="log-modal-body">
                    <pre>${JSON.stringify(logEntry, null, 2)}</pre>
                </div>
            </div>
        `;
        modal.addEventListener('click', (e) => {
            if (e.target === modal) modal.remove();
        });
        document.body.appendChild(modal);
    }

    filterByLevel(level) {
        this.filters.level = level;
        this.applyFilters();
    }

    filterByAgent(agent) {
        this.filters.agent = agent;
        this.applyFilters();
    }

    filterByTool(tool) {
        this.filters.tool = tool;
        this.applyFilters();
    }

    filterBySearch(search) {
        this.filters.search = search;
        this.applyFilters();
    }

    applyFilters() {
        this.filteredLogs = this.logs.filter(log => this.matchesFilters(log));
        this.renderAllLogs();
    }

    renderAllLogs() {
        this.logContent.innerHTML = '';

        if (this.filteredLogs.length === 0) {
            this.logContent.innerHTML = '<div class="log-empty">No matching logs</div>';
        } else {
            this.filteredLogs.forEach(log => this.renderLogEntry(log));
        }

        this.updateCount();

        if (this.autoScroll) {
            this.scrollToBottom();
        }
    }

    toggleAutoScroll(enabled) {
        this.autoScroll = enabled;
        if (enabled) {
            this.scrollToBottom();
        }
    }

    scrollToBottom() {
        this.logContent.scrollTop = this.logContent.scrollHeight;
    }

    clear() {
        this.logs = [];
        this.filteredLogs = [];
        this.logContent.innerHTML = '<div class="log-empty">Logs cleared</div>';
        this.updateCount();
    }

    updateCount() {
        const countEl = document.getElementById('logCount');
        const filtered = this.filteredLogs.length;
        const total = this.logs.length;
        countEl.textContent = filtered === total
            ? `${total} logs`
            : `${filtered} / ${total} logs`;
    }

    exportLogs() {
        const logsToExport = this.filteredLogs.length > 0 ? this.filteredLogs : this.logs;
        const content = logsToExport.map(log => {
            const parts = [
                log.timestamp,
                log.level,
                log.agent || '-',
                log.tool || '-',
                log.message
            ];
            return parts.join('\t');
        }).join('\n');

        const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
        const url = URL.createObjectURL(blob);

        const link = document.createElement('a');
        link.href = url;
        link.download = `ragix-logs-${new Date().toISOString().split('T')[0]}.log`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    }

    /**
     * Connect to WebSocket for real-time log streaming.
     */
    connectStream(wsUrl) {
        const statusEl = document.getElementById('logStreamStatus');

        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            statusEl.textContent = 'Connected';
            statusEl.className = 'log-stream-status log-stream-connected';
        };

        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (data.type === 'log') {
                    this.addLog(data.log);
                } else if (data.type === 'logs') {
                    this.addLogs(data.logs);
                }
            } catch (error) {
                console.error('Failed to parse log message:', error);
            }
        };

        this.ws.onclose = () => {
            statusEl.textContent = 'Disconnected';
            statusEl.className = 'log-stream-status log-stream-disconnected';
        };

        this.ws.onerror = (error) => {
            console.error('Log stream error:', error);
            statusEl.textContent = 'Error';
            statusEl.className = 'log-stream-status log-stream-error';
        };
    }

    disconnectStream() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }

    formatTime(timestamp) {
        if (!timestamp) return '';
        const date = new Date(timestamp);
        return date.toLocaleTimeString('en-US', { hour12: false }) +
               '.' + String(date.getMilliseconds()).padStart(3, '0');
    }

    escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Global instance
let logViewer = null;

function initLogViewer(containerId) {
    logViewer = new LogViewer(containerId);
    return logViewer;
}
