/**
 * WASP Runtime - WebAssembly-ready Agentic System Protocol Runtime
 *
 * Browser-side tool execution for RAGIX agents.
 * Provides a unified interface for executing tools client-side.
 *
 * Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-26
 */

class WaspRuntime {
    constructor(options = {}) {
        this.tools = new Map();
        this.modules = new Map();
        this.virtualFS = options.virtualFS || new VirtualFS();
        this.maxOutputSize = options.maxOutputSize || 50000;
        this.debug = options.debug || false;

        // Built-in tools
        this._registerBuiltinTools();
    }

    /**
     * Register built-in JavaScript tools
     */
    _registerBuiltinTools() {
        // JSON Validation
        this.registerTool('validate_json', {
            category: 'validation',
            description: 'Validate JSON content',
            execute: (inputs) => this._validateJson(inputs)
        });

        this.registerTool('format_json', {
            category: 'validation',
            description: 'Format/prettify JSON',
            execute: (inputs) => this._formatJson(inputs)
        });

        // Markdown tools
        this.registerTool('extract_headers', {
            category: 'markdown',
            description: 'Extract headers from Markdown',
            execute: (inputs) => this._extractHeaders(inputs)
        });

        this.registerTool('generate_toc', {
            category: 'markdown',
            description: 'Generate table of contents',
            execute: (inputs) => this._generateToc(inputs)
        });

        // Search tools
        this.registerTool('search_pattern', {
            category: 'search',
            description: 'Search for regex pattern',
            execute: (inputs) => this._searchPattern(inputs)
        });

        this.registerTool('count_matches', {
            category: 'search',
            description: 'Count pattern matches',
            execute: (inputs) => this._countMatches(inputs)
        });

        this.registerTool('replace_pattern', {
            category: 'search',
            description: 'Replace pattern matches',
            execute: (inputs) => this._replacePattern(inputs)
        });
    }

    /**
     * Register a tool
     */
    registerTool(name, config) {
        this.tools.set(name, {
            name,
            category: config.category || 'custom',
            description: config.description || '',
            execute: config.execute
        });
    }

    /**
     * Check if a tool is available
     */
    hasTool(name) {
        return this.tools.has(name);
    }

    /**
     * List available tools
     */
    listTools() {
        const tools = [];
        for (const [name, info] of this.tools) {
            tools.push({
                name,
                category: info.category,
                description: info.description
            });
        }
        return tools;
    }

    /**
     * Execute a tool
     */
    async execute(toolName, inputs = {}) {
        const startTime = performance.now();

        if (!this.tools.has(toolName)) {
            return {
                tool: toolName,
                success: false,
                error: `Unknown tool: ${toolName}`,
                duration_ms: performance.now() - startTime
            };
        }

        const tool = this.tools.get(toolName);

        try {
            const result = await tool.execute(inputs);
            const duration = performance.now() - startTime;

            // Check if result indicates failure
            let success = true;
            if (typeof result === 'object' && result !== null) {
                if (result.success === false || result.valid === false) {
                    success = false;
                }
            }

            return {
                tool: toolName,
                success,
                result,
                duration_ms: duration
            };

        } catch (error) {
            return {
                tool: toolName,
                success: false,
                error: error.message,
                duration_ms: performance.now() - startTime
            };
        }
    }

    /**
     * Execute a wasp_task action (compatible with RAGIX protocol)
     */
    async executeAction(action) {
        if (action.action !== 'wasp_task') {
            throw new Error(`Expected wasp_task action, got: ${action.action}`);
        }

        return this.execute(action.tool, action.inputs || {});
    }

    // ==================== Built-in Tool Implementations ====================

    _validateJson(inputs) {
        const { content, strict } = inputs;

        if (!content) {
            return { valid: false, error: 'No content provided' };
        }

        let processedContent = content;
        if (!strict) {
            // Remove comments and trailing commas
            processedContent = content
                .replace(/\/\/.*$/gm, '')
                .replace(/,(\s*[}\]])/g, '$1');
        }

        try {
            const data = JSON.parse(processedContent);
            const result = {
                valid: true,
                type: Array.isArray(data) ? 'array' : typeof data
            };

            if (Array.isArray(data)) {
                result.item_count = data.length;
            } else if (typeof data === 'object' && data !== null) {
                result.key_count = Object.keys(data).length;
                result.keys = Object.keys(data).slice(0, 20);
            }

            return result;

        } catch (e) {
            return {
                valid: false,
                error: e.message
            };
        }
    }

    _formatJson(inputs) {
        const { content, indent = 2, sort_keys = false, compact = false } = inputs;

        try {
            let data = JSON.parse(content);

            if (sort_keys && typeof data === 'object' && data !== null) {
                data = this._sortObjectKeys(data);
            }

            const formatted = compact
                ? JSON.stringify(data)
                : JSON.stringify(data, null, indent);

            return {
                success: true,
                formatted,
                original_size: content.length,
                formatted_size: formatted.length
            };

        } catch (e) {
            return {
                success: false,
                error: `Invalid JSON: ${e.message}`
            };
        }
    }

    _sortObjectKeys(obj) {
        if (Array.isArray(obj)) {
            return obj.map(item => this._sortObjectKeys(item));
        }
        if (typeof obj === 'object' && obj !== null) {
            const sorted = {};
            for (const key of Object.keys(obj).sort()) {
                sorted[key] = this._sortObjectKeys(obj[key]);
            }
            return sorted;
        }
        return obj;
    }

    _extractHeaders(inputs) {
        const { content } = inputs;
        const headers = [];
        const lines = content.split('\n');

        for (let i = 0; i < lines.length; i++) {
            const match = lines[i].match(/^(#{1,6})\s+(.+)$/);
            if (match) {
                headers.push({
                    level: match[1].length,
                    text: match[2].trim(),
                    line: i + 1
                });
            }
        }

        return {
            success: true,
            headers,
            count: headers.length
        };
    }

    _generateToc(inputs) {
        const { content, max_level = 3, bullet = '-' } = inputs;
        const headers = this._extractHeaders({ content }).headers;
        const entries = [];
        const tocLines = [];

        for (const header of headers) {
            if (header.level > max_level) continue;

            const anchor = header.text
                .toLowerCase()
                .replace(/[^\w\s-]/g, '')
                .replace(/[\s_]+/g, '-')
                .replace(/^-+|-+$/g, '');

            const indent = '  '.repeat(header.level - 1);
            tocLines.push(`${indent}${bullet} [${header.text}](#${anchor})`);

            entries.push({
                text: header.text,
                level: header.level,
                anchor
            });
        }

        return {
            success: true,
            toc: tocLines.join('\n'),
            entries
        };
    }

    _searchPattern(inputs) {
        const { pattern, content, flags = '', max_matches = 100 } = inputs;
        const matches = [];

        try {
            let regexFlags = 'g';
            if (flags.includes('i')) regexFlags += 'i';
            if (flags.includes('m')) regexFlags += 'm';
            if (flags.includes('s')) regexFlags += 's';

            const regex = new RegExp(pattern, regexFlags);
            const lines = content.split('\n');

            // Build line index
            const lineStarts = [0];
            for (let i = 0; i < content.length; i++) {
                if (content[i] === '\n') {
                    lineStarts.push(i + 1);
                }
            }

            let match;
            let count = 0;
            while ((match = regex.exec(content)) !== null && count < max_matches) {
                // Find line number
                let lineNum = 1;
                for (let i = 0; i < lineStarts.length; i++) {
                    if (match.index < lineStarts[i]) {
                        lineNum = i;
                        break;
                    }
                    lineNum = i + 1;
                }

                matches.push({
                    text: match[0],
                    start: match.index,
                    end: match.index + match[0].length,
                    line: lineNum,
                    groups: match.slice(1)
                });

                count++;

                // Prevent infinite loop for zero-length matches
                if (match.index === regex.lastIndex) {
                    regex.lastIndex++;
                }
            }

            return {
                success: true,
                matches,
                count: matches.length,
                truncated: count >= max_matches
            };

        } catch (e) {
            return {
                success: false,
                error: `Invalid regex: ${e.message}`,
                matches: [],
                count: 0
            };
        }
    }

    _countMatches(inputs) {
        const { pattern, content, flags = '' } = inputs;

        try {
            let regexFlags = 'g';
            if (flags.includes('i')) regexFlags += 'i';
            if (flags.includes('m')) regexFlags += 'm';

            const regex = new RegExp(pattern, regexFlags);
            const matches = content.match(regex) || [];

            // Count lines with matches
            const lines = content.split('\n');
            let linesWithMatches = 0;
            const lineRegex = new RegExp(pattern, regexFlags.replace('g', ''));
            for (const line of lines) {
                if (lineRegex.test(line)) {
                    linesWithMatches++;
                }
            }

            return {
                success: true,
                count: matches.length,
                line_count: linesWithMatches,
                total_lines: lines.length
            };

        } catch (e) {
            return {
                success: false,
                error: `Invalid regex: ${e.message}`,
                count: 0
            };
        }
    }

    _replacePattern(inputs) {
        const { pattern, replacement, content, flags = '', count = 0 } = inputs;

        try {
            let regexFlags = count === 0 ? 'g' : '';
            if (flags.includes('i')) regexFlags += 'i';
            if (flags.includes('m')) regexFlags += 'm';
            if (flags.includes('s')) regexFlags += 's';

            const regex = new RegExp(pattern, regexFlags);

            let result;
            let replacements = 0;

            if (count === 0) {
                // Replace all
                const matches = content.match(new RegExp(pattern, 'g' + regexFlags)) || [];
                replacements = matches.length;
                result = content.replace(regex, replacement);
            } else {
                // Replace limited count
                result = content;
                for (let i = 0; i < count; i++) {
                    const newResult = result.replace(regex, replacement);
                    if (newResult === result) break;
                    result = newResult;
                    replacements++;
                }
            }

            return {
                success: true,
                content: result,
                replacements,
                changed: replacements > 0
            };

        } catch (e) {
            return {
                success: false,
                error: `Invalid regex: ${e.message}`,
                content,
                replacements: 0
            };
        }
    }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { WaspRuntime };
}
