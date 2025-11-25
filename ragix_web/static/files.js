/**
 * RAGIX File Browser - Sandbox file tree and preview
 *
 * Provides a tree view of the sandbox directory with
 * file preview and basic operations.
 *
 * Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-25
 */

class FileBrowser {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.rootPath = '.';
        this.currentPath = '.';
        this.selectedFile = null;
        this.expandedDirs = new Set();

        // File type icons
        this.fileIcons = {
            'py': '🐍',
            'js': '📜',
            'ts': '📘',
            'json': '📋',
            'yaml': '📄',
            'yml': '📄',
            'md': '📝',
            'txt': '📄',
            'html': '🌐',
            'css': '🎨',
            'sh': '⚙️',
            'bash': '⚙️',
            'default': '📄',
            'folder': '📁',
            'folder-open': '📂'
        };

        this.init();
    }

    init() {
        this.container.innerHTML = `
            <div class="file-browser">
                <div class="file-toolbar">
                    <button class="file-btn" onclick="fileBrowser.refresh()" title="Refresh">↻</button>
                    <span class="file-path" id="fileBrowserPath">${this.rootPath}</span>
                    <button class="file-btn" onclick="fileBrowser.goUp()" title="Go Up">↑</button>
                </div>
                <div class="file-content">
                    <div class="file-tree" id="fileTree">
                        <div class="file-loading">Loading...</div>
                    </div>
                    <div class="file-preview" id="filePreview">
                        <div class="file-preview-header">
                            <span id="previewFileName">Select a file</span>
                        </div>
                        <div class="file-preview-content" id="previewContent">
                            <div class="file-preview-empty">Select a file to preview</div>
                        </div>
                    </div>
                </div>
            </div>
        `;

        this.fileTree = document.getElementById('fileTree');
        this.previewContent = document.getElementById('previewContent');
    }

    /**
     * Set the root path for the file browser.
     */
    setRoot(path) {
        this.rootPath = path;
        this.currentPath = path;
        document.getElementById('fileBrowserPath').textContent = path;
        this.refresh();
    }

    /**
     * Refresh the file tree.
     */
    async refresh() {
        this.fileTree.innerHTML = '<div class="file-loading">Loading...</div>';

        try {
            const response = await fetch(`/api/files?path=${encodeURIComponent(this.currentPath)}`);
            const data = await response.json();

            if (data.error) {
                this.fileTree.innerHTML = `<div class="file-error">${data.error}</div>`;
                return;
            }

            this.renderTree(data.files);
        } catch (error) {
            console.error('Failed to load files:', error);
            this.fileTree.innerHTML = '<div class="file-error">Failed to load files</div>';
        }
    }

    /**
     * Load a directory tree from data.
     */
    loadTree(treeData) {
        this.renderTree(treeData);
    }

    renderTree(files) {
        this.fileTree.innerHTML = '';

        if (!files || files.length === 0) {
            this.fileTree.innerHTML = '<div class="file-empty">Empty directory</div>';
            return;
        }

        // Sort: directories first, then files, alphabetically
        files.sort((a, b) => {
            if (a.isDirectory !== b.isDirectory) {
                return a.isDirectory ? -1 : 1;
            }
            return a.name.localeCompare(b.name);
        });

        files.forEach(file => {
            this.renderNode(file, this.fileTree, 0);
        });
    }

    renderNode(file, parent, depth) {
        const node = document.createElement('div');
        node.className = 'file-node';
        node.style.paddingLeft = `${depth * 16 + 8}px`;

        const icon = this.getIcon(file);
        const isExpanded = this.expandedDirs.has(file.path);

        if (file.isDirectory) {
            node.innerHTML = `
                <span class="file-expand">${isExpanded ? '▼' : '▶'}</span>
                <span class="file-icon">${icon}</span>
                <span class="file-name">${file.name}</span>
            `;
            node.classList.add('file-directory');

            node.addEventListener('click', (e) => {
                e.stopPropagation();
                this.toggleDirectory(file, node, depth);
            });

            if (isExpanded && file.children) {
                const childContainer = document.createElement('div');
                childContainer.className = 'file-children';
                file.children.forEach(child => {
                    this.renderNode(child, childContainer, depth + 1);
                });
                node.appendChild(childContainer);
            }
        } else {
            node.innerHTML = `
                <span class="file-expand"></span>
                <span class="file-icon">${icon}</span>
                <span class="file-name">${file.name}</span>
                <span class="file-size">${this.formatSize(file.size)}</span>
            `;
            node.classList.add('file-file');

            if (this.selectedFile === file.path) {
                node.classList.add('file-selected');
            }

            node.addEventListener('click', (e) => {
                e.stopPropagation();
                this.selectFile(file, node);
            });
        }

        parent.appendChild(node);
    }

    getIcon(file) {
        if (file.isDirectory) {
            return this.expandedDirs.has(file.path)
                ? this.fileIcons['folder-open']
                : this.fileIcons['folder'];
        }

        const ext = file.name.split('.').pop().toLowerCase();
        return this.fileIcons[ext] || this.fileIcons['default'];
    }

    async toggleDirectory(dir, node, depth) {
        if (this.expandedDirs.has(dir.path)) {
            // Collapse
            this.expandedDirs.delete(dir.path);
            const children = node.querySelector('.file-children');
            if (children) children.remove();
            node.querySelector('.file-expand').textContent = '▶';
            node.querySelector('.file-icon').textContent = this.fileIcons['folder'];
        } else {
            // Expand
            this.expandedDirs.add(dir.path);
            node.querySelector('.file-expand').textContent = '▼';
            node.querySelector('.file-icon').textContent = this.fileIcons['folder-open'];

            // Load children if not already loaded
            if (!dir.children) {
                try {
                    const response = await fetch(`/api/files?path=${encodeURIComponent(dir.path)}`);
                    const data = await response.json();
                    dir.children = data.files || [];
                } catch (error) {
                    console.error('Failed to load directory:', error);
                    dir.children = [];
                }
            }

            const childContainer = document.createElement('div');
            childContainer.className = 'file-children';

            dir.children.sort((a, b) => {
                if (a.isDirectory !== b.isDirectory) {
                    return a.isDirectory ? -1 : 1;
                }
                return a.name.localeCompare(b.name);
            });

            dir.children.forEach(child => {
                this.renderNode(child, childContainer, depth + 1);
            });

            node.appendChild(childContainer);
        }
    }

    async selectFile(file, node) {
        // Update selection
        const prev = this.fileTree.querySelector('.file-selected');
        if (prev) prev.classList.remove('file-selected');
        node.classList.add('file-selected');
        this.selectedFile = file.path;

        // Update preview header
        document.getElementById('previewFileName').textContent = file.name;

        // Load file content
        try {
            const response = await fetch(`/api/file?path=${encodeURIComponent(file.path)}`);
            const data = await response.json();

            if (data.error) {
                this.previewContent.innerHTML = `<div class="file-preview-error">${data.error}</div>`;
                return;
            }

            this.renderPreview(file, data.content);
        } catch (error) {
            console.error('Failed to load file:', error);
            this.previewContent.innerHTML = '<div class="file-preview-error">Failed to load file</div>';
        }
    }

    renderPreview(file, content) {
        const ext = file.name.split('.').pop().toLowerCase();
        const isText = ['py', 'js', 'ts', 'json', 'yaml', 'yml', 'md', 'txt', 'html', 'css', 'sh', 'bash', 'xml', 'toml', 'ini', 'cfg'].includes(ext);

        if (!isText || content.length > 100000) {
            this.previewContent.innerHTML = `
                <div class="file-preview-binary">
                    <p>File: ${file.name}</p>
                    <p>Size: ${this.formatSize(file.size)}</p>
                    <p>Type: ${ext.toUpperCase()}</p>
                    ${content.length > 100000 ? '<p>File too large to preview</p>' : ''}
                </div>
            `;
            return;
        }

        // Render with line numbers
        const lines = content.split('\n');
        let html = '<table class="file-preview-code"><tbody>';

        lines.forEach((line, i) => {
            html += `<tr>
                <td class="file-line-num">${i + 1}</td>
                <td class="file-line-content">${this.escapeHtml(line) || ' '}</td>
            </tr>`;
        });

        html += '</tbody></table>';
        this.previewContent.innerHTML = html;
    }

    goUp() {
        if (this.currentPath === this.rootPath || this.currentPath === '.') {
            return;
        }

        const parts = this.currentPath.split('/');
        parts.pop();
        this.currentPath = parts.join('/') || this.rootPath;
        document.getElementById('fileBrowserPath').textContent = this.currentPath;
        this.refresh();
    }

    formatSize(bytes) {
        if (bytes === undefined || bytes === null) return '';
        if (bytes < 1024) return `${bytes} B`;
        if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
        return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    }

    escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Global instance
let fileBrowser = null;

function initFileBrowser(containerId) {
    fileBrowser = new FileBrowser(containerId);
    return fileBrowser;
}
