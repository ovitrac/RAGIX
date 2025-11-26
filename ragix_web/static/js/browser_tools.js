/**
 * Browser Tools Integration - Connect WASP tools with RAGIX Web UI
 *
 * Provides UI components and integration for browser-side tool execution.
 *
 * Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-26
 */

class BrowserToolsUI {
    constructor(options = {}) {
        this.runtime = options.runtime || new WaspRuntime();
        this.virtualFS = options.virtualFS || new VirtualFS();
        this.containerId = options.containerId || 'browser-tools';
        this.onResult = options.onResult || null;

        // Execution mode: 'browser' or 'server'
        this.executionMode = options.executionMode || 'browser';

        // Server WebSocket for server-side execution
        this.serverWs = options.serverWs || null;
    }

    /**
     * Initialize the UI
     */
    init() {
        this._createUI();
        this._bindEvents();
        this._populateToolList();
    }

    /**
     * Create the browser tools UI
     */
    _createUI() {
        const container = document.getElementById(this.containerId);
        if (!container) {
            console.warn(`Container #${this.containerId} not found`);
            return;
        }

        container.innerHTML = `
            <div class="browser-tools-panel">
                <div class="tools-header">
                    <h3>WASP Tools</h3>
                    <div class="execution-mode">
                        <label>
                            <input type="radio" name="exec-mode" value="browser" checked>
                            Browser
                        </label>
                        <label>
                            <input type="radio" name="exec-mode" value="server">
                            Server
                        </label>
                    </div>
                </div>

                <div class="tool-selector">
                    <select id="tool-select">
                        <option value="">Select a tool...</option>
                    </select>
                    <div id="tool-description" class="tool-description"></div>
                </div>

                <div class="tool-inputs">
                    <div class="input-group">
                        <label for="tool-content">Content:</label>
                        <textarea id="tool-content" rows="8" placeholder="Enter content to process..."></textarea>
                    </div>

                    <div class="input-group" id="additional-inputs">
                        <!-- Dynamic inputs added here -->
                    </div>
                </div>

                <div class="tool-actions">
                    <button id="execute-tool" class="btn-primary">Execute</button>
                    <button id="clear-tool" class="btn-secondary">Clear</button>
                    <button id="load-file" class="btn-secondary">Load File</button>
                    <input type="file" id="file-input" style="display: none">
                </div>

                <div class="tool-result">
                    <h4>Result:</h4>
                    <div class="result-meta">
                        <span id="result-status"></span>
                        <span id="result-duration"></span>
                    </div>
                    <pre id="result-output"></pre>
                </div>
            </div>
        `;

        // Add styles
        this._addStyles();
    }

    /**
     * Add CSS styles for the UI
     */
    _addStyles() {
        if (document.getElementById('browser-tools-styles')) return;

        const styles = document.createElement('style');
        styles.id = 'browser-tools-styles';
        styles.textContent = `
            .browser-tools-panel {
                padding: 1rem;
                border: 1px solid #ddd;
                border-radius: 8px;
                background: #f9f9f9;
            }

            .tools-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 1rem;
            }

            .tools-header h3 {
                margin: 0;
            }

            .execution-mode label {
                margin-left: 1rem;
                cursor: pointer;
            }

            .tool-selector {
                margin-bottom: 1rem;
            }

            .tool-selector select {
                width: 100%;
                padding: 0.5rem;
                font-size: 1rem;
                border: 1px solid #ccc;
                border-radius: 4px;
            }

            .tool-description {
                margin-top: 0.5rem;
                font-size: 0.9rem;
                color: #666;
                font-style: italic;
            }

            .tool-inputs .input-group {
                margin-bottom: 1rem;
            }

            .tool-inputs label {
                display: block;
                margin-bottom: 0.25rem;
                font-weight: bold;
            }

            .tool-inputs textarea,
            .tool-inputs input[type="text"] {
                width: 100%;
                padding: 0.5rem;
                font-family: monospace;
                font-size: 0.9rem;
                border: 1px solid #ccc;
                border-radius: 4px;
                box-sizing: border-box;
            }

            .tool-actions {
                margin-bottom: 1rem;
            }

            .tool-actions button {
                padding: 0.5rem 1rem;
                margin-right: 0.5rem;
                font-size: 1rem;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }

            .btn-primary {
                background: #4a90d9;
                color: white;
            }

            .btn-primary:hover {
                background: #357abd;
            }

            .btn-secondary {
                background: #6c757d;
                color: white;
            }

            .btn-secondary:hover {
                background: #5a6268;
            }

            .tool-result {
                background: white;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 1rem;
            }

            .tool-result h4 {
                margin: 0 0 0.5rem 0;
            }

            .result-meta {
                display: flex;
                gap: 1rem;
                margin-bottom: 0.5rem;
                font-size: 0.9rem;
            }

            .result-meta .success {
                color: #28a745;
            }

            .result-meta .error {
                color: #dc3545;
            }

            #result-output {
                background: #f4f4f4;
                padding: 1rem;
                border-radius: 4px;
                overflow-x: auto;
                max-height: 400px;
                margin: 0;
                font-size: 0.85rem;
            }
        `;
        document.head.appendChild(styles);
    }

    /**
     * Bind event handlers
     */
    _bindEvents() {
        // Tool selection
        const toolSelect = document.getElementById('tool-select');
        if (toolSelect) {
            toolSelect.addEventListener('change', (e) => {
                this._onToolSelect(e.target.value);
            });
        }

        // Execute button
        const executeBtn = document.getElementById('execute-tool');
        if (executeBtn) {
            executeBtn.addEventListener('click', () => {
                this._executeCurrentTool();
            });
        }

        // Clear button
        const clearBtn = document.getElementById('clear-tool');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => {
                this._clearInputs();
            });
        }

        // Load file button
        const loadBtn = document.getElementById('load-file');
        const fileInput = document.getElementById('file-input');
        if (loadBtn && fileInput) {
            loadBtn.addEventListener('click', () => {
                fileInput.click();
            });
            fileInput.addEventListener('change', (e) => {
                this._loadFile(e.target.files[0]);
            });
        }

        // Execution mode toggle
        const modeInputs = document.querySelectorAll('input[name="exec-mode"]');
        modeInputs.forEach(input => {
            input.addEventListener('change', (e) => {
                this.executionMode = e.target.value;
            });
        });
    }

    /**
     * Populate the tool selection dropdown
     */
    _populateToolList() {
        const select = document.getElementById('tool-select');
        if (!select) return;

        const tools = this.runtime.listTools();

        // Group by category
        const byCategory = {};
        for (const tool of tools) {
            const cat = tool.category || 'other';
            if (!byCategory[cat]) {
                byCategory[cat] = [];
            }
            byCategory[cat].push(tool);
        }

        // Create optgroups
        for (const [category, categoryTools] of Object.entries(byCategory).sort()) {
            const group = document.createElement('optgroup');
            group.label = category.charAt(0).toUpperCase() + category.slice(1);

            for (const tool of categoryTools.sort((a, b) => a.name.localeCompare(b.name))) {
                const option = document.createElement('option');
                option.value = tool.name;
                option.textContent = tool.name;
                group.appendChild(option);
            }

            select.appendChild(group);
        }
    }

    /**
     * Handle tool selection
     */
    _onToolSelect(toolName) {
        const descEl = document.getElementById('tool-description');

        if (!toolName) {
            if (descEl) descEl.textContent = '';
            return;
        }

        const tools = this.runtime.listTools();
        const tool = tools.find(t => t.name === toolName);

        if (descEl && tool) {
            descEl.textContent = tool.description || 'No description available';
        }
    }

    /**
     * Execute the currently selected tool
     */
    async _executeCurrentTool() {
        const toolName = document.getElementById('tool-select')?.value;
        const content = document.getElementById('tool-content')?.value;

        if (!toolName) {
            this._showResult({ success: false, error: 'Please select a tool' });
            return;
        }

        const inputs = { content };

        // Collect additional inputs
        const additionalInputs = document.querySelectorAll('#additional-inputs input');
        additionalInputs.forEach(input => {
            if (input.value) {
                inputs[input.name] = input.value;
            }
        });

        // Execute based on mode
        let result;
        if (this.executionMode === 'browser') {
            result = await this.runtime.execute(toolName, inputs);
        } else {
            result = await this._executeOnServer(toolName, inputs);
        }

        this._showResult(result);

        if (this.onResult) {
            this.onResult(result);
        }
    }

    /**
     * Execute tool on server via WebSocket
     */
    async _executeOnServer(toolName, inputs) {
        if (!this.serverWs || this.serverWs.readyState !== WebSocket.OPEN) {
            return {
                success: false,
                error: 'Server connection not available'
            };
        }

        return new Promise((resolve) => {
            const action = {
                action: 'wasp_task',
                tool: toolName,
                inputs
            };

            // Set up response handler
            const handler = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    if (data.type === 'wasp_result') {
                        this.serverWs.removeEventListener('message', handler);
                        resolve(data.result);
                    }
                } catch (e) {
                    // Ignore non-JSON messages
                }
            };

            this.serverWs.addEventListener('message', handler);
            this.serverWs.send(JSON.stringify({ type: 'wasp_task', action }));

            // Timeout after 30 seconds
            setTimeout(() => {
                this.serverWs.removeEventListener('message', handler);
                resolve({ success: false, error: 'Server timeout' });
            }, 30000);
        });
    }

    /**
     * Show execution result
     */
    _showResult(result) {
        const statusEl = document.getElementById('result-status');
        const durationEl = document.getElementById('result-duration');
        const outputEl = document.getElementById('result-output');

        if (statusEl) {
            statusEl.textContent = result.success ? 'Success' : 'Error';
            statusEl.className = result.success ? 'success' : 'error';
        }

        if (durationEl && result.duration_ms !== undefined) {
            durationEl.textContent = `(${result.duration_ms.toFixed(1)}ms)`;
        }

        if (outputEl) {
            if (result.error) {
                outputEl.textContent = `Error: ${result.error}`;
            } else {
                outputEl.textContent = JSON.stringify(result.result, null, 2);
            }
        }
    }

    /**
     * Clear input fields
     */
    _clearInputs() {
        const content = document.getElementById('tool-content');
        if (content) content.value = '';

        const output = document.getElementById('result-output');
        if (output) output.textContent = '';

        const status = document.getElementById('result-status');
        if (status) status.textContent = '';

        const duration = document.getElementById('result-duration');
        if (duration) duration.textContent = '';
    }

    /**
     * Load content from file
     */
    async _loadFile(file) {
        if (!file) return;

        try {
            const content = await file.text();
            const contentEl = document.getElementById('tool-content');
            if (contentEl) {
                contentEl.value = content;
            }
        } catch (error) {
            this._showResult({ success: false, error: `Failed to load file: ${error.message}` });
        }
    }

    /**
     * Set server WebSocket connection
     */
    setServerConnection(ws) {
        this.serverWs = ws;
    }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { BrowserToolsUI };
}
