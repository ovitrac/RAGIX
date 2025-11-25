/**
 * RAGIX Workflow Visualizer - Interactive graph visualization
 *
 * Uses D3.js for force-directed graph rendering with real-time
 * status updates and interactive node inspection.
 *
 * Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-25
 */

class WorkflowVisualizer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.width = 600;
        this.height = 400;
        this.nodes = [];
        this.edges = [];
        this.simulation = null;
        this.svg = null;
        this.nodeGroup = null;
        this.edgeGroup = null;
        this.selectedNode = null;

        // Status colors
        this.statusColors = {
            'pending': '#6b7280',
            'running': '#f59e0b',
            'completed': '#10b981',
            'failed': '#ef4444',
            'skipped': '#9ca3af'
        };

        // Agent type icons (Unicode)
        this.agentIcons = {
            'code': '\u2699',     // Gear
            'doc': '\u270F',      // Pencil
            'test': '\u2713',     // Check mark
            'git': '\u2387',      // Branch
            'default': '\u25CF'   // Circle
        };

        this.init();
    }

    init() {
        // Create SVG container
        this.svg = d3.select(this.container)
            .append('svg')
            .attr('width', '100%')
            .attr('height', '100%')
            .attr('viewBox', `0 0 ${this.width} ${this.height}`)
            .attr('class', 'workflow-svg');

        // Add arrow marker for edges
        this.svg.append('defs')
            .append('marker')
            .attr('id', 'arrowhead')
            .attr('viewBox', '-0 -5 10 10')
            .attr('refX', 20)
            .attr('refY', 0)
            .attr('orient', 'auto')
            .attr('markerWidth', 6)
            .attr('markerHeight', 6)
            .append('path')
            .attr('d', 'M 0,-5 L 10,0 L 0,5')
            .attr('fill', '#6b7280');

        // Create groups for edges and nodes
        this.edgeGroup = this.svg.append('g').attr('class', 'edges');
        this.nodeGroup = this.svg.append('g').attr('class', 'nodes');

        // Initialize force simulation
        this.simulation = d3.forceSimulation()
            .force('link', d3.forceLink().id(d => d.id).distance(100))
            .force('charge', d3.forceManyBody().strength(-300))
            .force('center', d3.forceCenter(this.width / 2, this.height / 2))
            .force('collision', d3.forceCollide().radius(40));

        // Add zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.5, 3])
            .on('zoom', (event) => {
                this.nodeGroup.attr('transform', event.transform);
                this.edgeGroup.attr('transform', event.transform);
            });

        this.svg.call(zoom);

        // Create info panel
        this.createInfoPanel();
    }

    createInfoPanel() {
        const panel = document.createElement('div');
        panel.id = 'workflowInfo';
        panel.className = 'workflow-info hidden';
        panel.innerHTML = `
            <div class="workflow-info-header">
                <span class="workflow-info-title">Node Details</span>
                <button onclick="workflowViz.closeInfo()" class="workflow-info-close">&times;</button>
            </div>
            <div class="workflow-info-content">
                <div class="workflow-info-field">
                    <label>Name:</label>
                    <span id="infoNodeName">-</span>
                </div>
                <div class="workflow-info-field">
                    <label>Type:</label>
                    <span id="infoNodeType">-</span>
                </div>
                <div class="workflow-info-field">
                    <label>Status:</label>
                    <span id="infoNodeStatus">-</span>
                </div>
                <div class="workflow-info-field">
                    <label>Task:</label>
                    <div id="infoNodeTask" class="workflow-info-task">-</div>
                </div>
                <div class="workflow-info-field">
                    <label>Tools:</label>
                    <span id="infoNodeTools">-</span>
                </div>
            </div>
        `;
        this.container.appendChild(panel);
        this.infoPanel = panel;
    }

    loadWorkflow(graphData) {
        // Convert graph data to D3 format
        this.nodes = Object.values(graphData.nodes || {}).map(node => ({
            id: node.node_id,
            name: node.node_id,
            type: node.agent_type,
            task: node.task,
            tools: node.allowed_tools || [],
            status: node.status || 'pending',
            x: Math.random() * this.width,
            y: Math.random() * this.height
        }));

        this.edges = (graphData.edges || []).map(edge => ({
            source: edge.source,
            target: edge.target,
            condition: edge.condition
        }));

        this.render();
    }

    render() {
        // Clear existing elements
        this.edgeGroup.selectAll('*').remove();
        this.nodeGroup.selectAll('*').remove();

        // Draw edges
        const links = this.edgeGroup.selectAll('line')
            .data(this.edges)
            .enter()
            .append('line')
            .attr('class', 'workflow-edge')
            .attr('stroke', '#6b7280')
            .attr('stroke-width', 2)
            .attr('marker-end', 'url(#arrowhead)');

        // Draw nodes
        const nodeGroups = this.nodeGroup.selectAll('g')
            .data(this.nodes)
            .enter()
            .append('g')
            .attr('class', 'workflow-node')
            .call(d3.drag()
                .on('start', (event, d) => this.dragStarted(event, d))
                .on('drag', (event, d) => this.dragged(event, d))
                .on('end', (event, d) => this.dragEnded(event, d)))
            .on('click', (event, d) => this.selectNode(d));

        // Node circles
        nodeGroups.append('circle')
            .attr('r', 25)
            .attr('fill', d => this.statusColors[d.status] || this.statusColors['pending'])
            .attr('stroke', '#1f2937')
            .attr('stroke-width', 2);

        // Node icons
        nodeGroups.append('text')
            .attr('class', 'workflow-node-icon')
            .attr('text-anchor', 'middle')
            .attr('dominant-baseline', 'middle')
            .attr('fill', 'white')
            .attr('font-size', '14px')
            .text(d => this.agentIcons[d.type] || this.agentIcons['default']);

        // Node labels
        nodeGroups.append('text')
            .attr('class', 'workflow-node-label')
            .attr('text-anchor', 'middle')
            .attr('dy', 40)
            .attr('fill', '#e5e7eb')
            .attr('font-size', '11px')
            .text(d => d.name);

        // Update simulation
        this.simulation
            .nodes(this.nodes)
            .on('tick', () => this.tick(links, nodeGroups));

        this.simulation.force('link').links(this.edges);
        this.simulation.alpha(1).restart();
    }

    tick(links, nodeGroups) {
        links
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y);

        nodeGroups
            .attr('transform', d => `translate(${d.x}, ${d.y})`);
    }

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

    selectNode(node) {
        this.selectedNode = node;

        // Update visual selection
        this.nodeGroup.selectAll('circle')
            .attr('stroke-width', d => d.id === node.id ? 4 : 2)
            .attr('stroke', d => d.id === node.id ? '#3b82f6' : '#1f2937');

        // Update info panel
        document.getElementById('infoNodeName').textContent = node.name;
        document.getElementById('infoNodeType').textContent = node.type;
        document.getElementById('infoNodeStatus').textContent = node.status;
        document.getElementById('infoNodeTask').textContent = node.task;
        document.getElementById('infoNodeTools').textContent = node.tools.join(', ') || 'None';

        // Show info panel
        this.infoPanel.classList.remove('hidden');
    }

    closeInfo() {
        this.infoPanel.classList.add('hidden');
        this.selectedNode = null;

        // Clear visual selection
        this.nodeGroup.selectAll('circle')
            .attr('stroke-width', 2)
            .attr('stroke', '#1f2937');
    }

    updateNodeStatus(nodeId, status) {
        const node = this.nodes.find(n => n.id === nodeId);
        if (node) {
            node.status = status;

            this.nodeGroup.selectAll('circle')
                .filter(d => d.id === nodeId)
                .transition()
                .duration(300)
                .attr('fill', this.statusColors[status] || this.statusColors['pending']);

            // Update info panel if this node is selected
            if (this.selectedNode && this.selectedNode.id === nodeId) {
                document.getElementById('infoNodeStatus').textContent = status;
            }
        }
    }

    setLayout(layout) {
        switch (layout) {
            case 'horizontal':
                this.horizontalLayout();
                break;
            case 'vertical':
                this.verticalLayout();
                break;
            case 'force':
            default:
                this.simulation.alpha(1).restart();
        }
    }

    horizontalLayout() {
        const padding = 80;
        const nodeWidth = (this.width - padding * 2) / (this.nodes.length || 1);

        this.nodes.forEach((node, i) => {
            node.fx = padding + i * nodeWidth + nodeWidth / 2;
            node.fy = this.height / 2;
        });

        this.simulation.alpha(0.3).restart();
    }

    verticalLayout() {
        const padding = 60;
        const nodeHeight = (this.height - padding * 2) / (this.nodes.length || 1);

        this.nodes.forEach((node, i) => {
            node.fx = this.width / 2;
            node.fy = padding + i * nodeHeight + nodeHeight / 2;
        });

        this.simulation.alpha(0.3).restart();
    }

    resetLayout() {
        this.nodes.forEach(node => {
            node.fx = null;
            node.fy = null;
        });
        this.simulation.alpha(1).restart();
    }

    exportSVG() {
        const svgData = this.svg.node().outerHTML;
        const blob = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' });
        const url = URL.createObjectURL(blob);

        const link = document.createElement('a');
        link.href = url;
        link.download = 'workflow.svg';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    }
}

// Global instance
let workflowViz = null;

function initWorkflowVisualizer(containerId) {
    workflowViz = new WorkflowVisualizer(containerId);
    return workflowViz;
}
