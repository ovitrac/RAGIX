/**
 * Virtual Filesystem - In-memory filesystem for browser-side WASP tools
 *
 * Provides file operations for browser-based tool execution.
 * Can integrate with File System Access API for local file access.
 *
 * Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-26
 */

class VirtualFS {
    constructor(options = {}) {
        this.files = new Map();
        this.directories = new Set(['/']);
        this.maxFileSize = options.maxFileSize || 10 * 1024 * 1024; // 10MB default
        this.maxFiles = options.maxFiles || 1000;
    }

    /**
     * Normalize a path
     */
    _normalizePath(path) {
        // Ensure path starts with /
        if (!path.startsWith('/')) {
            path = '/' + path;
        }
        // Remove trailing slash except for root
        if (path !== '/' && path.endsWith('/')) {
            path = path.slice(0, -1);
        }
        // Resolve . and ..
        const parts = path.split('/').filter(p => p && p !== '.');
        const resolved = [];
        for (const part of parts) {
            if (part === '..') {
                resolved.pop();
            } else {
                resolved.push(part);
            }
        }
        return '/' + resolved.join('/');
    }

    /**
     * Get parent directory of a path
     */
    _getParentDir(path) {
        const normalized = this._normalizePath(path);
        const lastSlash = normalized.lastIndexOf('/');
        return lastSlash === 0 ? '/' : normalized.slice(0, lastSlash);
    }

    /**
     * Check if a path exists
     */
    exists(path) {
        const normalized = this._normalizePath(path);
        return this.files.has(normalized) || this.directories.has(normalized);
    }

    /**
     * Check if path is a file
     */
    isFile(path) {
        return this.files.has(this._normalizePath(path));
    }

    /**
     * Check if path is a directory
     */
    isDirectory(path) {
        return this.directories.has(this._normalizePath(path));
    }

    /**
     * Read a file
     */
    readFile(path, encoding = 'utf-8') {
        const normalized = this._normalizePath(path);

        if (!this.files.has(normalized)) {
            throw new Error(`File not found: ${path}`);
        }

        const file = this.files.get(normalized);

        if (encoding === 'utf-8' || encoding === 'text') {
            if (typeof file.content === 'string') {
                return file.content;
            }
            // Decode ArrayBuffer to string
            const decoder = new TextDecoder('utf-8');
            return decoder.decode(file.content);
        }

        return file.content;
    }

    /**
     * Write a file
     */
    writeFile(path, content) {
        const normalized = this._normalizePath(path);

        // Check limits
        if (this.files.size >= this.maxFiles && !this.files.has(normalized)) {
            throw new Error(`Maximum file limit reached: ${this.maxFiles}`);
        }

        const size = typeof content === 'string'
            ? new Blob([content]).size
            : content.byteLength || content.length;

        if (size > this.maxFileSize) {
            throw new Error(`File too large: ${size} > ${this.maxFileSize}`);
        }

        // Ensure parent directory exists
        const parentDir = this._getParentDir(normalized);
        this._ensureDirectory(parentDir);

        this.files.set(normalized, {
            content,
            size,
            modified: Date.now(),
            created: this.files.has(normalized)
                ? this.files.get(normalized).created
                : Date.now()
        });

        return { success: true, path: normalized, size };
    }

    /**
     * Delete a file
     */
    deleteFile(path) {
        const normalized = this._normalizePath(path);

        if (!this.files.has(normalized)) {
            return { success: false, error: 'File not found' };
        }

        this.files.delete(normalized);
        return { success: true };
    }

    /**
     * Ensure a directory exists (create if needed)
     */
    _ensureDirectory(path) {
        const normalized = this._normalizePath(path);
        if (normalized === '/') return;

        const parts = normalized.split('/').filter(Boolean);
        let current = '';

        for (const part of parts) {
            current += '/' + part;
            this.directories.add(current);
        }
    }

    /**
     * Create a directory
     */
    mkdir(path, recursive = false) {
        const normalized = this._normalizePath(path);

        if (this.files.has(normalized)) {
            throw new Error(`A file exists at path: ${path}`);
        }

        if (recursive) {
            this._ensureDirectory(normalized);
        } else {
            const parentDir = this._getParentDir(normalized);
            if (!this.directories.has(parentDir)) {
                throw new Error(`Parent directory does not exist: ${parentDir}`);
            }
            this.directories.add(normalized);
        }

        return { success: true, path: normalized };
    }

    /**
     * List directory contents
     */
    listDir(path = '/') {
        const normalized = this._normalizePath(path);

        if (!this.directories.has(normalized)) {
            throw new Error(`Directory not found: ${path}`);
        }

        const prefix = normalized === '/' ? '/' : normalized + '/';
        const entries = [];

        // Find files in directory
        for (const [filePath, file] of this.files) {
            if (filePath.startsWith(prefix)) {
                const relativePath = filePath.slice(prefix.length);
                if (!relativePath.includes('/')) {
                    entries.push({
                        name: relativePath,
                        type: 'file',
                        size: file.size,
                        modified: file.modified
                    });
                }
            }
        }

        // Find subdirectories
        for (const dirPath of this.directories) {
            if (dirPath.startsWith(prefix) && dirPath !== normalized) {
                const relativePath = dirPath.slice(prefix.length);
                if (!relativePath.includes('/')) {
                    entries.push({
                        name: relativePath,
                        type: 'directory'
                    });
                }
            }
        }

        return entries.sort((a, b) => {
            if (a.type !== b.type) {
                return a.type === 'directory' ? -1 : 1;
            }
            return a.name.localeCompare(b.name);
        });
    }

    /**
     * Get file stats
     */
    stat(path) {
        const normalized = this._normalizePath(path);

        if (this.files.has(normalized)) {
            const file = this.files.get(normalized);
            return {
                type: 'file',
                size: file.size,
                created: file.created,
                modified: file.modified
            };
        }

        if (this.directories.has(normalized)) {
            return {
                type: 'directory'
            };
        }

        throw new Error(`Path not found: ${path}`);
    }

    /**
     * Copy a file
     */
    copyFile(src, dest) {
        const content = this.readFile(src);
        return this.writeFile(dest, content);
    }

    /**
     * Move/rename a file
     */
    moveFile(src, dest) {
        const content = this.readFile(src);
        this.writeFile(dest, content);
        this.deleteFile(src);
        return { success: true, from: src, to: dest };
    }

    /**
     * Get filesystem statistics
     */
    getStats() {
        let totalSize = 0;
        for (const file of this.files.values()) {
            totalSize += file.size;
        }

        return {
            fileCount: this.files.size,
            directoryCount: this.directories.size,
            totalSize,
            maxFiles: this.maxFiles,
            maxFileSize: this.maxFileSize
        };
    }

    /**
     * Clear all files and directories
     */
    clear() {
        this.files.clear();
        this.directories.clear();
        this.directories.add('/');
        return { success: true };
    }

    /**
     * Load files from a FileList (drag-drop or file input)
     */
    async loadFromFileList(fileList, basePath = '/') {
        const results = [];

        for (const file of fileList) {
            try {
                const content = await file.text();
                const path = this._normalizePath(basePath + '/' + file.name);
                this.writeFile(path, content);
                results.push({ path, success: true, size: content.length });
            } catch (error) {
                results.push({ name: file.name, success: false, error: error.message });
            }
        }

        return results;
    }

    /**
     * Load directory using File System Access API
     * Only works in supported browsers with user permission
     */
    async loadFromDirectoryHandle(dirHandle, basePath = '/') {
        const results = [];

        async function* getFilesRecursively(entry, path) {
            if (entry.kind === 'file') {
                const file = await entry.getFile();
                yield { path: path + '/' + entry.name, file };
            } else if (entry.kind === 'directory') {
                for await (const subEntry of entry.values()) {
                    yield* getFilesRecursively(subEntry, path + '/' + entry.name);
                }
            }
        }

        try {
            for await (const { path, file } of getFilesRecursively(dirHandle, basePath)) {
                try {
                    const content = await file.text();
                    const normalizedPath = this._normalizePath(path);
                    this.writeFile(normalizedPath, content);
                    results.push({ path: normalizedPath, success: true, size: content.length });
                } catch (error) {
                    results.push({ path, success: false, error: error.message });
                }
            }
        } catch (error) {
            results.push({ error: `Failed to read directory: ${error.message}` });
        }

        return results;
    }

    /**
     * Export filesystem as a blob (for download)
     */
    async exportAsZip() {
        // Simple implementation without external dependencies
        // Returns an array of file objects that can be used with a zip library
        const files = [];

        for (const [path, file] of this.files) {
            files.push({
                path: path.slice(1), // Remove leading /
                content: file.content,
                modified: new Date(file.modified)
            });
        }

        return files;
    }

    /**
     * Export a single file as downloadable blob
     */
    exportFile(path) {
        const content = this.readFile(path);
        const blob = new Blob([content], { type: 'text/plain' });
        const normalized = this._normalizePath(path);
        const filename = normalized.split('/').pop();

        return { blob, filename };
    }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { VirtualFS };
}
