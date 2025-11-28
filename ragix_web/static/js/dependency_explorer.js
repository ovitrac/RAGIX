/**
 * Dependency Explorer - Interactive force-directed graph visualization
 *
 * Features:
 * - Package clustering with force-directed layout
 * - Edge bundling for cleaner visualization
 * - Zoom/pan navigation
 * - Search and filter
 * - Click to highlight dependencies
 * - Export to SVG/PNG
 *
 * Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-27
 */

class DependencyExplorer {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            throw new Error(`Container not found: ${containerId}`);
        }

        // Default options
        this.options = {
            width: options.width || this.container.clientWidth || 1200,
            height: options.height || this.container.clientHeight || 800,
            nodeRadius: options.nodeRadius || 8,
            linkDistance: options.linkDistance || 100,
            chargeStrength: options.chargeStrength || -300,
            clusterStrength: options.clusterStrength || 0.5,
            showLabels: options.showLabels !== false,
            showMinimap: options.showMinimap !== false,
            edgeBundling: options.edgeBundling !== false,
            colorScheme: options.colorScheme || 'default',
            ...options
        };

        // Color schemes
        this.colorSchemes = {
            default: {
                class: '#4a90d9',
                interface: '#50c878',
                method: '#f5a623',
                function: '#f5a623',
                field: '#9b59b6',
                constant: '#e74c3c',
                module: '#34495e',
                package: '#2c3e50',
                enum: '#1abc9c'
            },
            pastel: {
                class: '#aed6f1',
                interface: '#abebc6',
                method: '#fdebd0',
                function: '#fdebd0',
                field: '#d7bde2',
                constant: '#f5b7b1',
                module: '#d5dbdb',
                package: '#bdc3c7',
                enum: '#a3e4d7'
            },
            dark: {
                class: '#2980b9',
                interface: '#27ae60',
                method: '#d35400',
                function: '#d35400',
                field: '#8e44ad',
                constant: '#c0392b',
                module: '#2c3e50',
                package: '#1a252f',
                enum: '#16a085'
            }
        };

        // Edge colors by type
        this.edgeColors = {
            import: '#999999',
            inheritance: '#3498db',
            implementation: '#27ae60',
            call: '#2ecc71',
            composition: '#9b59b6',
            annotation: '#e74c3c',
            type_reference: '#95a5a6'
        };

        // State
        this.data = null;
        this.nodes = [];
        this.links = [];
        this.clusters = new Map();
        this.selectedNode = null;
        this.highlightedNodes = new Set();
        this.filterTypes = new Set();
        this.filterDepTypes = new Set();
        this.searchTerm = '';

        this.init();
    }

    init() {
        // Create main SVG
        this.svg = d3.select(this.container)
            .append('svg')
            .attr('width', this.options.width)
            .attr('height', this.options.height)
            .attr('class', 'dependency-explorer');

        // Add zoom behavior
        this.zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on('zoom', (event) => this.handleZoom(event));

        this.svg.call(this.zoom);

        // Create main group for graph elements
        this.g = this.svg.append('g').attr('class', 'graph-content');

        // Create layers
        this.linksLayer = this.g.append('g').attr('class', 'links-layer');
        this.nodesLayer = this.g.append('g').attr('class', 'nodes-layer');
        this.labelsLayer = this.g.append('g').attr('class', 'labels-layer');
        this.clustersLayer = this.g.insert('g', '.links-layer').attr('class', 'clusters-layer');

        // Create tooltip
        this.tooltip = d3.select(this.container)
            .append('div')
            .attr('class', 'dep-tooltip')
            .style('opacity', 0)
            .style('position', 'absolute')
            .style('background', '#2c3e50')
            .style('color', 'white')
            .style('padding', '10px')
            .style('border-radius', '4px')
            .style('font-size', '12px')
            .style('pointer-events', 'none')
            .style('z-index', '1000');

        // Create minimap if enabled
        if (this.options.showMinimap) {
            this.createMinimap();
        }

        // Arrow markers for directed edges
        this.createArrowMarkers();

        // Initialize simulation
        this.simulation = d3.forceSimulation()
            .force('link', d3.forceLink()
                .id(d => d.id)
                .distance(this.options.linkDistance))
            .force('charge', d3.forceManyBody()
                .strength(this.options.chargeStrength))
            .force('center', d3.forceCenter(
                this.options.width / 2,
                this.options.height / 2))
            .force('collision', d3.forceCollide()
                .radius(this.options.nodeRadius * 2))
            .on('tick', () => this.tick());
    }

    createArrowMarkers() {
        const defs = this.svg.append('defs');

        Object.entries(this.edgeColors).forEach(([type, color]) => {
            defs.append('marker')
                .attr('id', `arrow-${type}`)
                .attr('viewBox', '0 -5 10 10')
                .attr('refX', 15)
                .attr('refY', 0)
                .attr('markerWidth', 6)
                .attr('markerHeight', 6)
                .attr('orient', 'auto')
                .append('path')
                .attr('d', 'M0,-5L10,0L0,5')
                .attr('fill', color);
        });
    }

    createMinimap() {
        const minimapSize = 150;
        const padding = 10;

        this.minimap = this.svg.append('g')
            .attr('class', 'minimap')
            .attr('transform', `translate(${this.options.width - minimapSize - padding}, ${padding})`);

        this.minimap.append('rect')
            .attr('width', minimapSize)
            .attr('height', minimapSize)
            .attr('fill', '#f5f5f5')
            .attr('stroke', '#ccc')
            .attr('rx', 4);

        this.minimapContent = this.minimap.append('g');
        this.minimapViewport = this.minimap.append('rect')
            .attr('class', 'minimap-viewport')
            .attr('fill', 'rgba(74, 144, 217, 0.3)')
            .attr('stroke', '#4a90d9')
            .attr('stroke-width', 2);
    }

    loadData(data) {
        this.data = data;
        this.nodes = data.nodes.map(d => ({ ...d }));
        this.links = data.links.map(d => ({ ...d }));

        // Build cluster information from package names
        this.buildClusters();

        // Update simulation
        this.simulation.nodes(this.nodes);
        this.simulation.force('link').links(this.links);

        // Add cluster force
        this.simulation.force('cluster', this.clusterForce());

        // Render graph
        this.render();

        // Restart simulation
        this.simulation.alpha(1).restart();
    }

    buildClusters() {
        this.clusters.clear();

        this.nodes.forEach(node => {
            // Extract package from qualified name
            const parts = node.id.split('.');
            const packageName = parts.length > 2 ?
                parts.slice(0, -1).join('.') :
                parts[0];

            if (!this.clusters.has(packageName)) {
                this.clusters.set(packageName, {
                    name: packageName,
                    nodes: [],
                    x: Math.random() * this.options.width,
                    y: Math.random() * this.options.height,
                    color: this.getClusterColor(packageName)
                });
            }

            const cluster = this.clusters.get(packageName);
            cluster.nodes.push(node);
            node.cluster = cluster;
        });
    }

    getClusterColor(packageName) {
        // Generate consistent color from package name
        let hash = 0;
        for (let i = 0; i < packageName.length; i++) {
            hash = packageName.charCodeAt(i) + ((hash << 5) - hash);
        }
        const hue = Math.abs(hash % 360);
        return `hsl(${hue}, 60%, 85%)`;
    }

    clusterForce() {
        const strength = this.options.clusterStrength;
        const nodes = this.nodes;

        return (alpha) => {
            nodes.forEach(node => {
                if (node.cluster) {
                    node.vx -= (node.x - node.cluster.x) * strength * alpha;
                    node.vy -= (node.y - node.cluster.y) * strength * alpha;
                }
            });
        };
    }

    render() {
        this.renderClusters();
        this.renderLinks();
        this.renderNodes();
        if (this.options.showLabels) {
            this.renderLabels();
        }
        if (this.options.showMinimap) {
            this.updateMinimap();
        }
    }

    renderClusters() {
        const clusterData = Array.from(this.clusters.values());

        // Compute cluster hulls
        const clusterHulls = this.clustersLayer.selectAll('.cluster-hull')
            .data(clusterData, d => d.name);

        clusterHulls.exit().remove();

        clusterHulls.enter()
            .append('path')
            .attr('class', 'cluster-hull')
            .merge(clusterHulls)
            .attr('fill', d => d.color)
            .attr('stroke', d => d3.color(d.color).darker(0.5))
            .attr('stroke-width', 2)
            .attr('opacity', 0.3);
    }

    updateClusterHulls() {
        this.clustersLayer.selectAll('.cluster-hull')
            .attr('d', d => {
                const points = d.nodes.map(n => [n.x, n.y]);
                if (points.length < 3) {
                    // Draw circle for small clusters
                    const cx = d3.mean(points, p => p[0]) || 0;
                    const cy = d3.mean(points, p => p[1]) || 0;
                    return `M${cx-30},${cy} a30,30 0 1,0 60,0 a30,30 0 1,0 -60,0`;
                }
                const hull = d3.polygonHull(points);
                if (!hull) return '';
                return 'M' + hull.map(p => p.join(',')).join('L') + 'Z';
            });
    }

    renderLinks() {
        const linkData = this.getFilteredLinks();

        const links = this.linksLayer.selectAll('.link')
            .data(linkData, d => `${d.source.id || d.source}-${d.target.id || d.target}`);

        links.exit().remove();

        const newLinks = links.enter()
            .append('path')
            .attr('class', 'link')
            .attr('fill', 'none')
            .attr('stroke-opacity', 0.6)
            .on('mouseover', (event, d) => this.showLinkTooltip(event, d))
            .on('mouseout', () => this.hideTooltip());

        newLinks.merge(links)
            .attr('stroke', d => this.edgeColors[d.type] || '#999')
            .attr('stroke-width', d => d.type === 'inheritance' ? 2 : 1)
            .attr('stroke-dasharray', d => {
                if (d.type === 'import') return '5,5';
                if (d.type === 'implementation') return '3,3';
                return null;
            })
            .attr('marker-end', d => `url(#arrow-${d.type || 'call'})`);
    }

    renderNodes() {
        const nodeData = this.getFilteredNodes();

        const nodes = this.nodesLayer.selectAll('.node')
            .data(nodeData, d => d.id);

        nodes.exit().remove();

        const newNodes = nodes.enter()
            .append('circle')
            .attr('class', 'node')
            .attr('r', this.options.nodeRadius)
            .call(d3.drag()
                .on('start', (event, d) => this.dragStarted(event, d))
                .on('drag', (event, d) => this.dragged(event, d))
                .on('end', (event, d) => this.dragEnded(event, d)))
            .on('click', (event, d) => this.selectNode(event, d))
            .on('dblclick', (event, d) => this.expandNode(event, d))
            .on('mouseover', (event, d) => this.showNodeTooltip(event, d))
            .on('mouseout', () => this.hideTooltip());

        newNodes.merge(nodes)
            .attr('fill', d => this.getNodeColor(d))
            .attr('stroke', d => this.selectedNode === d ? '#333' : 'white')
            .attr('stroke-width', d => this.selectedNode === d ? 3 : 1.5)
            .style('cursor', 'pointer')
            .style('opacity', d => this.isNodeVisible(d) ? 1 : 0.2);
    }

    renderLabels() {
        const nodeData = this.getFilteredNodes();

        const labels = this.labelsLayer.selectAll('.node-label')
            .data(nodeData, d => d.id);

        labels.exit().remove();

        const newLabels = labels.enter()
            .append('text')
            .attr('class', 'node-label')
            .attr('font-size', 10)
            .attr('dx', 12)
            .attr('dy', 4)
            .style('pointer-events', 'none');

        newLabels.merge(labels)
            .text(d => this.truncate(d.name, 20))
            .style('opacity', d => this.isNodeVisible(d) ? 1 : 0.2);
    }

    tick() {
        // Update cluster centers
        this.clusters.forEach(cluster => {
            cluster.x = d3.mean(cluster.nodes, n => n.x) || cluster.x;
            cluster.y = d3.mean(cluster.nodes, n => n.y) || cluster.y;
        });

        // Update cluster hulls
        this.updateClusterHulls();

        // Update links
        if (this.options.edgeBundling) {
            this.linksLayer.selectAll('.link')
                .attr('d', d => this.bundledEdge(d));
        } else {
            this.linksLayer.selectAll('.link')
                .attr('d', d => `M${d.source.x},${d.source.y}L${d.target.x},${d.target.y}`);
        }

        // Update nodes
        this.nodesLayer.selectAll('.node')
            .attr('cx', d => d.x)
            .attr('cy', d => d.y);

        // Update labels
        this.labelsLayer.selectAll('.node-label')
            .attr('x', d => d.x)
            .attr('y', d => d.y);

        // Update minimap
        if (this.options.showMinimap) {
            this.updateMinimap();
        }
    }

    bundledEdge(d) {
        // Simple quadratic curve through cluster centers for edge bundling effect
        const sx = d.source.x;
        const sy = d.source.y;
        const tx = d.target.x;
        const ty = d.target.y;

        if (d.source.cluster === d.target.cluster) {
            // Same cluster: straight line
            return `M${sx},${sy}L${tx},${ty}`;
        }

        // Different clusters: curve through midpoint biased toward center
        const mx = (sx + tx) / 2;
        const my = (sy + ty) / 2;
        const cx = this.options.width / 2;
        const cy = this.options.height / 2;
        const controlX = mx + (cx - mx) * 0.3;
        const controlY = my + (cy - my) * 0.3;

        return `M${sx},${sy}Q${controlX},${controlY} ${tx},${ty}`;
    }

    handleZoom(event) {
        this.g.attr('transform', event.transform);
        if (this.options.showMinimap) {
            this.updateMinimapViewport(event.transform);
        }
    }

    updateMinimap() {
        const minimapScale = 0.1;

        // Update minimap content
        const minimapNodes = this.minimapContent.selectAll('.minimap-node')
            .data(this.nodes, d => d.id);

        minimapNodes.exit().remove();

        minimapNodes.enter()
            .append('circle')
            .attr('class', 'minimap-node')
            .attr('r', 2)
            .merge(minimapNodes)
            .attr('cx', d => d.x * minimapScale)
            .attr('cy', d => d.y * minimapScale)
            .attr('fill', d => this.getNodeColor(d));
    }

    updateMinimapViewport(transform) {
        const minimapScale = 0.1;
        const viewportX = -transform.x / transform.k * minimapScale;
        const viewportY = -transform.y / transform.k * minimapScale;
        const viewportWidth = this.options.width / transform.k * minimapScale;
        const viewportHeight = this.options.height / transform.k * minimapScale;

        this.minimapViewport
            .attr('x', viewportX)
            .attr('y', viewportY)
            .attr('width', viewportWidth)
            .attr('height', viewportHeight);
    }

    // Interaction methods
    dragStarted(event, d) {
        if (!event.active) this.simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }

    dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }

    dragEnded(event, d) {
        if (!event.active) this.simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }

    selectNode(event, d) {
        event.stopPropagation();

        if (this.selectedNode === d) {
            this.selectedNode = null;
            this.highlightedNodes.clear();
        } else {
            this.selectedNode = d;
            this.highlightConnectedNodes(d);
        }

        this.renderNodes();
        this.renderLabels();
        this.dispatchEvent('nodeSelected', d);
    }

    highlightConnectedNodes(node) {
        this.highlightedNodes.clear();
        this.highlightedNodes.add(node.id);

        // Find all connected nodes
        this.links.forEach(link => {
            const sourceId = link.source.id || link.source;
            const targetId = link.target.id || link.target;

            if (sourceId === node.id) {
                this.highlightedNodes.add(targetId);
            }
            if (targetId === node.id) {
                this.highlightedNodes.add(sourceId);
            }
        });
    }

    expandNode(event, d) {
        // Double-click: toggle cluster expansion
        if (d.cluster) {
            d.cluster.expanded = !d.cluster.expanded;
            this.simulation.alpha(0.3).restart();
        }
        this.dispatchEvent('nodeExpanded', d);
    }

    showNodeTooltip(event, d) {
        const inDeps = this.links.filter(l => (l.target.id || l.target) === d.id).length;
        const outDeps = this.links.filter(l => (l.source.id || l.source) === d.id).length;

        this.tooltip
            .style('opacity', 1)
            .style('left', (event.pageX + 10) + 'px')
            .style('top', (event.pageY - 10) + 'px')
            .html(`
                <strong>${d.name}</strong><br>
                <small>${d.type}</small><br>
                <hr style="border-color: #555; margin: 5px 0;">
                In: ${inDeps} | Out: ${outDeps}<br>
                ${d.file ? `File: ${d.file.split('/').pop()}` : ''}
                ${d.line ? `<br>Line: ${d.line}` : ''}
            `);
    }

    showLinkTooltip(event, d) {
        this.tooltip
            .style('opacity', 1)
            .style('left', (event.pageX + 10) + 'px')
            .style('top', (event.pageY - 10) + 'px')
            .html(`
                <strong>${d.type}</strong><br>
                ${(d.source.name || d.source)} → ${(d.target.name || d.target)}
            `);
    }

    hideTooltip() {
        this.tooltip.style('opacity', 0);
    }

    // Filtering
    filterByNodeType(types) {
        this.filterTypes = new Set(types);
        this.render();
    }

    filterByDepType(types) {
        this.filterDepTypes = new Set(types);
        this.render();
    }

    search(term) {
        this.searchTerm = term.toLowerCase();
        this.render();
    }

    getFilteredNodes() {
        return this.nodes.filter(d => {
            if (this.filterTypes.size > 0 && !this.filterTypes.has(d.type)) {
                return false;
            }
            return true;
        });
    }

    getFilteredLinks() {
        const nodeIds = new Set(this.getFilteredNodes().map(d => d.id));
        return this.links.filter(d => {
            const sourceId = d.source.id || d.source;
            const targetId = d.target.id || d.target;

            if (!nodeIds.has(sourceId) || !nodeIds.has(targetId)) {
                return false;
            }
            if (this.filterDepTypes.size > 0 && !this.filterDepTypes.has(d.type)) {
                return false;
            }
            return true;
        });
    }

    isNodeVisible(node) {
        if (this.searchTerm && !node.name.toLowerCase().includes(this.searchTerm)) {
            return false;
        }
        if (this.highlightedNodes.size > 0 && !this.highlightedNodes.has(node.id)) {
            return false;
        }
        return true;
    }

    // Helpers
    getNodeColor(node) {
        const colors = this.colorSchemes[this.options.colorScheme] || this.colorSchemes.default;
        return colors[node.type] || '#cccccc';
    }

    truncate(text, maxLength) {
        if (text.length <= maxLength) return text;
        return text.substring(0, maxLength - 3) + '...';
    }

    // Export
    exportSVG() {
        const svgData = new XMLSerializer().serializeToString(this.svg.node());
        const blob = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' });
        return URL.createObjectURL(blob);
    }

    exportPNG(callback) {
        const svgData = new XMLSerializer().serializeToString(this.svg.node());
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const img = new Image();

        canvas.width = this.options.width;
        canvas.height = this.options.height;

        img.onload = () => {
            ctx.fillStyle = 'white';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0);
            callback(canvas.toDataURL('image/png'));
        };

        img.src = 'data:image/svg+xml;base64,' + btoa(unescape(encodeURIComponent(svgData)));
    }

    // Zoom controls
    zoomIn() {
        this.svg.transition().call(this.zoom.scaleBy, 1.3);
    }

    zoomOut() {
        this.svg.transition().call(this.zoom.scaleBy, 0.7);
    }

    resetZoom() {
        this.svg.transition().call(this.zoom.transform, d3.zoomIdentity);
    }

    fitToView() {
        if (this.nodes.length === 0) return;

        const bounds = {
            minX: d3.min(this.nodes, d => d.x) - 50,
            maxX: d3.max(this.nodes, d => d.x) + 50,
            minY: d3.min(this.nodes, d => d.y) - 50,
            maxY: d3.max(this.nodes, d => d.y) + 50
        };

        const width = bounds.maxX - bounds.minX;
        const height = bounds.maxY - bounds.minY;
        const scale = 0.9 / Math.max(width / this.options.width, height / this.options.height);
        const translateX = (this.options.width - scale * (bounds.minX + bounds.maxX)) / 2;
        const translateY = (this.options.height - scale * (bounds.minY + bounds.maxY)) / 2;

        this.svg.transition().duration(750)
            .call(this.zoom.transform, d3.zoomIdentity
                .translate(translateX, translateY)
                .scale(scale));
    }

    // Event handling
    dispatchEvent(name, data) {
        this.container.dispatchEvent(new CustomEvent(name, { detail: data }));
    }

    on(eventName, callback) {
        this.container.addEventListener(eventName, callback);
    }

    // Cleanup
    destroy() {
        this.simulation.stop();
        this.svg.remove();
        this.tooltip.remove();
    }
}

// Export for use in browser and Node.js
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DependencyExplorer;
}
