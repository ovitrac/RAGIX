/**
 * RAGIX Diff Viewer - Side-by-side and inline diff display
 *
 * Provides syntax-highlighted diff viewing with accept/reject
 * functionality for code changes made by agents.
 *
 * Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-25
 */

class DiffViewer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.viewMode = 'split';  // 'split' or 'unified'
        this.diffs = [];
        this.currentIndex = 0;

        this.init();
    }

    init() {
        this.container.innerHTML = `
            <div class="diff-viewer">
                <div class="diff-toolbar">
                    <div class="diff-toolbar-left">
                        <span class="diff-file-name" id="diffFileName">No file selected</span>
                        <span class="diff-stats" id="diffStats"></span>
                    </div>
                    <div class="diff-toolbar-right">
                        <button class="diff-btn" onclick="diffViewer.toggleViewMode()">
                            <span id="viewModeIcon">⊟</span> Toggle View
                        </button>
                        <button class="diff-btn diff-btn-prev" onclick="diffViewer.prevDiff()" title="Previous">◀</button>
                        <span class="diff-nav" id="diffNav">0 / 0</span>
                        <button class="diff-btn diff-btn-next" onclick="diffViewer.nextDiff()" title="Next">▶</button>
                    </div>
                </div>
                <div class="diff-content" id="diffContent">
                    <div class="diff-empty">No diff to display</div>
                </div>
                <div class="diff-actions" id="diffActions">
                    <button class="diff-action-btn diff-accept" onclick="diffViewer.acceptChange()">
                        ✓ Accept Change
                    </button>
                    <button class="diff-action-btn diff-reject" onclick="diffViewer.rejectChange()">
                        ✕ Reject Change
                    </button>
                    <button class="diff-action-btn diff-copy" onclick="diffViewer.copyNewContent()">
                        ⧉ Copy New
                    </button>
                </div>
            </div>
        `;

        this.diffContent = document.getElementById('diffContent');
        this.diffActions = document.getElementById('diffActions');
        this.diffActions.style.display = 'none';
    }

    /**
     * Load a diff for display.
     *
     * @param {Object} diffData - Diff data object
     * @param {string} diffData.filePath - Path to the file
     * @param {string} diffData.oldContent - Original content
     * @param {string} diffData.newContent - New content
     * @param {string} diffData.diffText - Unified diff text (optional)
     */
    loadDiff(diffData) {
        this.diffs = [diffData];
        this.currentIndex = 0;
        this.renderCurrentDiff();
    }

    /**
     * Load multiple diffs.
     *
     * @param {Array} diffsArray - Array of diff data objects
     */
    loadDiffs(diffsArray) {
        this.diffs = diffsArray;
        this.currentIndex = 0;
        this.updateNav();
        if (this.diffs.length > 0) {
            this.renderCurrentDiff();
        }
    }

    renderCurrentDiff() {
        if (this.diffs.length === 0) {
            this.diffContent.innerHTML = '<div class="diff-empty">No diff to display</div>';
            this.diffActions.style.display = 'none';
            return;
        }

        const diff = this.diffs[this.currentIndex];

        // Update file name
        document.getElementById('diffFileName').textContent = diff.filePath || 'Unknown file';

        // Parse diff
        const parsed = this.parseDiff(diff.oldContent, diff.newContent);

        // Update stats
        document.getElementById('diffStats').textContent =
            `+${parsed.additions} -${parsed.deletions}`;

        // Render based on view mode
        if (this.viewMode === 'split') {
            this.renderSplitView(parsed);
        } else {
            this.renderUnifiedView(parsed);
        }

        this.diffActions.style.display = 'flex';
    }

    parseDiff(oldContent, newContent) {
        const oldLines = (oldContent || '').split('\n');
        const newLines = (newContent || '').split('\n');

        // Simple line-by-line diff using longest common subsequence
        const diff = this.computeDiff(oldLines, newLines);

        let additions = 0;
        let deletions = 0;

        diff.forEach(item => {
            if (item.type === 'add') additions++;
            if (item.type === 'remove') deletions++;
        });

        return {
            items: diff,
            additions,
            deletions,
            oldLines,
            newLines
        };
    }

    computeDiff(oldLines, newLines) {
        // Simple diff algorithm - compare lines
        const result = [];
        let oldIdx = 0;
        let newIdx = 0;

        while (oldIdx < oldLines.length || newIdx < newLines.length) {
            if (oldIdx >= oldLines.length) {
                // Rest of new lines are additions
                result.push({
                    type: 'add',
                    oldLine: null,
                    newLine: newIdx + 1,
                    content: newLines[newIdx]
                });
                newIdx++;
            } else if (newIdx >= newLines.length) {
                // Rest of old lines are deletions
                result.push({
                    type: 'remove',
                    oldLine: oldIdx + 1,
                    newLine: null,
                    content: oldLines[oldIdx]
                });
                oldIdx++;
            } else if (oldLines[oldIdx] === newLines[newIdx]) {
                // Same line
                result.push({
                    type: 'same',
                    oldLine: oldIdx + 1,
                    newLine: newIdx + 1,
                    content: oldLines[oldIdx]
                });
                oldIdx++;
                newIdx++;
            } else {
                // Look ahead to find if this is a change, add, or remove
                const oldInNew = newLines.indexOf(oldLines[oldIdx], newIdx);
                const newInOld = oldLines.indexOf(newLines[newIdx], oldIdx);

                if (oldInNew === -1 && newInOld === -1) {
                    // Both lines are different - treat as change
                    result.push({
                        type: 'remove',
                        oldLine: oldIdx + 1,
                        newLine: null,
                        content: oldLines[oldIdx]
                    });
                    result.push({
                        type: 'add',
                        oldLine: null,
                        newLine: newIdx + 1,
                        content: newLines[newIdx]
                    });
                    oldIdx++;
                    newIdx++;
                } else if (oldInNew !== -1 && (newInOld === -1 || oldInNew - newIdx < newInOld - oldIdx)) {
                    // New line inserted
                    result.push({
                        type: 'add',
                        oldLine: null,
                        newLine: newIdx + 1,
                        content: newLines[newIdx]
                    });
                    newIdx++;
                } else {
                    // Old line removed
                    result.push({
                        type: 'remove',
                        oldLine: oldIdx + 1,
                        newLine: null,
                        content: oldLines[oldIdx]
                    });
                    oldIdx++;
                }
            }
        }

        return result;
    }

    renderSplitView(parsed) {
        let html = '<div class="diff-split">';
        html += '<div class="diff-pane diff-old"><div class="diff-pane-header">Original</div>';
        html += '<div class="diff-lines">';

        // Left pane (old content)
        parsed.items.forEach(item => {
            if (item.type === 'same' || item.type === 'remove') {
                const cls = item.type === 'remove' ? 'diff-line-remove' : 'diff-line-same';
                html += `<div class="diff-line ${cls}">
                    <span class="diff-line-num">${item.oldLine || ''}</span>
                    <span class="diff-line-content">${this.escapeHtml(item.content)}</span>
                </div>`;
            } else if (item.type === 'add') {
                html += `<div class="diff-line diff-line-empty">
                    <span class="diff-line-num"></span>
                    <span class="diff-line-content"></span>
                </div>`;
            }
        });

        html += '</div></div>';
        html += '<div class="diff-pane diff-new"><div class="diff-pane-header">Modified</div>';
        html += '<div class="diff-lines">';

        // Right pane (new content)
        parsed.items.forEach(item => {
            if (item.type === 'same' || item.type === 'add') {
                const cls = item.type === 'add' ? 'diff-line-add' : 'diff-line-same';
                html += `<div class="diff-line ${cls}">
                    <span class="diff-line-num">${item.newLine || ''}</span>
                    <span class="diff-line-content">${this.escapeHtml(item.content)}</span>
                </div>`;
            } else if (item.type === 'remove') {
                html += `<div class="diff-line diff-line-empty">
                    <span class="diff-line-num"></span>
                    <span class="diff-line-content"></span>
                </div>`;
            }
        });

        html += '</div></div></div>';
        this.diffContent.innerHTML = html;
    }

    renderUnifiedView(parsed) {
        let html = '<div class="diff-unified"><div class="diff-lines">';

        parsed.items.forEach(item => {
            let prefix = ' ';
            let cls = 'diff-line-same';

            if (item.type === 'add') {
                prefix = '+';
                cls = 'diff-line-add';
            } else if (item.type === 'remove') {
                prefix = '-';
                cls = 'diff-line-remove';
            }

            html += `<div class="diff-line ${cls}">
                <span class="diff-line-num">${item.oldLine || ''}</span>
                <span class="diff-line-num">${item.newLine || ''}</span>
                <span class="diff-line-prefix">${prefix}</span>
                <span class="diff-line-content">${this.escapeHtml(item.content)}</span>
            </div>`;
        });

        html += '</div></div>';
        this.diffContent.innerHTML = html;
    }

    toggleViewMode() {
        this.viewMode = this.viewMode === 'split' ? 'unified' : 'split';
        document.getElementById('viewModeIcon').textContent = this.viewMode === 'split' ? '⊟' : '⊞';
        this.renderCurrentDiff();
    }

    prevDiff() {
        if (this.currentIndex > 0) {
            this.currentIndex--;
            this.updateNav();
            this.renderCurrentDiff();
        }
    }

    nextDiff() {
        if (this.currentIndex < this.diffs.length - 1) {
            this.currentIndex++;
            this.updateNav();
            this.renderCurrentDiff();
        }
    }

    updateNav() {
        document.getElementById('diffNav').textContent =
            `${this.diffs.length > 0 ? this.currentIndex + 1 : 0} / ${this.diffs.length}`;
    }

    acceptChange() {
        if (this.diffs.length === 0) return;

        const diff = this.diffs[this.currentIndex];
        const event = new CustomEvent('diffAccepted', {
            detail: {
                filePath: diff.filePath,
                newContent: diff.newContent
            }
        });
        this.container.dispatchEvent(event);

        // Remove from list
        this.diffs.splice(this.currentIndex, 1);
        if (this.currentIndex >= this.diffs.length) {
            this.currentIndex = Math.max(0, this.diffs.length - 1);
        }
        this.updateNav();
        this.renderCurrentDiff();
    }

    rejectChange() {
        if (this.diffs.length === 0) return;

        const diff = this.diffs[this.currentIndex];
        const event = new CustomEvent('diffRejected', {
            detail: {
                filePath: diff.filePath
            }
        });
        this.container.dispatchEvent(event);

        // Remove from list
        this.diffs.splice(this.currentIndex, 1);
        if (this.currentIndex >= this.diffs.length) {
            this.currentIndex = Math.max(0, this.diffs.length - 1);
        }
        this.updateNav();
        this.renderCurrentDiff();
    }

    copyNewContent() {
        if (this.diffs.length === 0) return;

        const diff = this.diffs[this.currentIndex];
        navigator.clipboard.writeText(diff.newContent || '').then(() => {
            // Show feedback
            const btn = this.container.querySelector('.diff-copy');
            const original = btn.textContent;
            btn.textContent = '✓ Copied!';
            setTimeout(() => {
                btn.textContent = original;
            }, 1500);
        });
    }

    clear() {
        this.diffs = [];
        this.currentIndex = 0;
        this.updateNav();
        this.diffContent.innerHTML = '<div class="diff-empty">No diff to display</div>';
        this.diffActions.style.display = 'none';
    }

    escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Global instance
let diffViewer = null;

function initDiffViewer(containerId) {
    diffViewer = new DiffViewer(containerId);
    return diffViewer;
}
