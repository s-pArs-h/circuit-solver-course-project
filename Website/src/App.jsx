import { useState, useRef } from 'react';
import './App.css';

const GRID_SIZE = 16;
const SNAP_RADIUS = GRID_SIZE;

// --- Component options for Edges ---
const componentOptions = [
  { id: 'wire', name: 'Wire', color: 'rgb(31 41 55)' },
  { id: 'resistor', name: 'Resistor', color: '#ef4444' }, // Red-500
  { id: 'voltage_source', name: 'Voltage Source', color: '#22c55e' }, // Green-500
  { id: 'capacitor', name: 'Capacitor', color: '#3b82f6' }, // Blue-500
  { id: 'inductor', name: 'Inductor', color: '#f97316' } // Orange-500
];

/**
 * Helper function to get component details by id.
 */
const getComponentById = (id) => {
  return componentOptions.find(c => c.id === id) || componentOptions[0]; // Default to wire
};

// --- SVG Path definitions for components ---
const ICON_LEN = 32; // Fixed pixel length of the icon part
const componentPaths = {
  // 32px long zigzag, centered at (0,0)
  resistor: "M -16 0 l 4 -6 l 8 12 l 8 -12 l 8 12 l 4 -6", 
  // 32px long parallel plates, centered at (0,0)
  capacitor: "M -16 0 L -2 0 M 2 0 L 16 0 M -2 -8 L -2 8 M 2 -8 L 2 8", 
  // 32px long coils, centered at (0,0)
  inductor: "M -16 0 q 4 10 8 0 q 4 -10 8 0 q 4 10 8 0 q 4 -10 8 0",
  // 32px long DC source, + on right (x2 side), - on left (x1 side)
  voltage_source: "M -16 0 L -6 0 M -6 -12 L -6 12 M 6 -6 L 6 6 M 6 0 L 16 0" 
};

/**
 * --- Component to render the SVG for a circuit element ---
 */
function CircuitComponent({ x1, y1, x2, y2, component }) {
  // Fixed wire thickness
  const wireStrokeWidth = 2; 
  // Fixed component icon thickness
  const componentIconStrokeWidth = 2; 

  // Wires are just simple lines with fixed thickness
  if (component.id === 'wire') {
    return (
      <line
        x1={x1} y1={y1}
        x2={x2} y2={y2}
        stroke={component.color}
        strokeWidth={wireStrokeWidth}
        style={{ pointerEvents: 'none' }} 
      />
    );
  }

  // --- Logic for all other components (resistor, cap, etc.) ---
  const dx = x2 - x1;
  const dy = y2 - y1;
  const L = Math.sqrt(dx * dx + dy * dy);
  const angle = Math.atan2(dy, dx) * 180 / Math.PI;
  
  const midX = (x1 + x2) / 2;
  const midY = (y1 + y2) / 2;
  
  // 'value' is no longer used for thickness, but we still need color/path
  const color = component.color;
  const pathD = componentPaths[component.id];

  // If the line is too short, just draw the icon scaled down
  if (L < ICON_LEN) {
    return (
      <g transform={`translate(${midX}, ${midY}) rotate(${angle}) scale(${L / ICON_LEN})`}>
        <path 
          d={pathD} 
          stroke={color} 
          // Counter-scale stroke width so it remains consistent
          strokeWidth={componentIconStrokeWidth / (L / ICON_LEN)} 
          fill="none" 
          style={{ pointerEvents: 'none' }} 
        />
      </g>
    );
  }

  // Line is long enough, draw wire-icon-wire
  const ux = dx / L;
  const uy = dy / L;
  const wireLen = (L - ICON_LEN) / 2;
  
  const iconStartX = x1 + ux * wireLen;
  const iconStartY = y1 + uy * wireLen;
  const iconEndX = x2 - ux * wireLen;
  const iconEndY = y2 - uy * wireLen;

  return (
    <g style={{ pointerEvents: 'none' }}>
      {/* Wire 1 */}
      <line x1={x1} y1={y1} x2={iconStartX} y2={iconStartY} stroke={color} strokeWidth={wireStrokeWidth} />
      
      {/* Icon */}
      <g transform={`translate(${midX}, ${midY}) rotate(${angle})`}>
        <path 
          d={pathD} 
          stroke={color} 
          strokeWidth={componentIconStrokeWidth}
          fill="none" 
        />
      </g>
      
      {/* Wire 2 */}
      <line x1={iconEndX} y1={iconEndY} x2={x2} y2={y2} stroke={color} strokeWidth={wireStrokeWidth} />
    </g>
  );
}


function App() {
  const [lines, setLines] = useState([]);
  const [nodes, setNodes] = useState([]);
  const [startPoint, setStartPoint] = useState(null);
  const [currentPos, setCurrentPos] = useState(null);
  const [showData, setShowData] = useState(false);
  const [copySuccess, setCopySuccess] = useState('');
  
  const [selectedComponentId, setSelectedComponentId] = useState(componentOptions[0].id); // 'wire'
  const [showComponentPanel, setShowComponentPanel] = useState(false);
  const [selectedLineInfo, setSelectedLineInfo] = useState(null); // { index: number, x: number, y: number }

  const [editingValue, setEditingValue] = useState(10);
  const [showImportPanel, setShowImportPanel] = useState(false);
  const [importJson, setImportJson] = useState('');
  const [importError, setImportError] = useState('');

  const mainRef = useRef(null);
  const dataTextRef = useRef(null);

  /**
   * Helper function to add a new node.
   */
  const addNode = (newPos) => {
    setNodes(prevNodes => {
      const exists = prevNodes.some(n => n.x === newPos.x && n.y === newPos.y);
      if (!exists) {
        return [...prevNodes, { ...newPos }];
      }
      return prevNodes;
    });
  };

  /**
   * Helper function to find the nearest grid point or snap target.
   */
  const getClickCoords = (e) => {
    if (!mainRef.current) return null;

    const rect = mainRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Check if near an existing node
    for (const node of nodes) {
      const dx = x - node.x;
      const dy = y - node.y;
      const distance = Math.sqrt(dx * dx + dy * dy);
      if (distance <= SNAP_RADIUS) {
        return { x: node.x, y: node.y }; // Snap to existing node
      }
    }

    // Check if near a grid point
    const nearestGridX = Math.round(x / GRID_SIZE) * GRID_SIZE;
    const nearestGridY = Math.round(y / GRID_SIZE) * GRID_SIZE;
    const dx = x - nearestGridX;
    const dy = y - nearestGridY;
    const distance = Math.sqrt(dx * dx + dy * dy);

    if (distance <= SNAP_RADIUS) {
      return { x: nearestGridX, y: nearestGridY }; // Snap to grid
    } else {
      return { x, y }; // Use exact position
    }
  };

  /**
   * Handles clicking to start or end a line.
   */
  const handleClick = (e) => {
    if (selectedLineInfo) {
      setSelectedLineInfo(null);
    }
    if (e.target.closest('.panel') || e.target.closest('.top-controls') || e.target.closest('.line-popup')) {
      return;
    }
    const coords = getClickCoords(e);
    if (!coords) return; 

    if (!startPoint) {
      setStartPoint(coords);
      addNode(coords); 
      setCurrentPos(coords);
    } else {
      addNode(coords); 
      
      const dx = coords.x - startPoint.x;
      const dy = coords.y - startPoint.y;
      const length = Math.sqrt(dx * dx + dy * dy);

      if (length > 0) {
        // Wires default to value 2 (for thickness), other components default to 10
        const newValue = selectedComponentId === 'wire' ? 2 : 10; 
        setLines(prevLines => [
          ...prevLines,
          { 
            x1: startPoint.x, 
            y1: startPoint.y, 
            x2: coords.x, 
            y2: coords.y, 
            component: selectedComponentId, 
            value: newValue                 
          }
        ]);
      } else {
        const isNodeConnected = (node, linesList) => {
          return linesList.some(l => 
            (l.x1 === node.x && l.y1 === node.y) || (l.x2 === node.x && l.y2 === node.y)
          );
        };
        if (!isNodeConnected(startPoint, lines)) {
          setNodes(prevNodes => prevNodes.filter(n => n.x !== startPoint.x || n.y !== startPoint.y));
        }
      }
      setStartPoint(null);
      setCurrentPos(null);
    }
  };

  /**
   * Handles mouse movement for preview line.
   */
  const handleMouseMove = (e) => {
    if (!mainRef.current || !startPoint) return; 
    const rect = mainRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setCurrentPos({ x, y });
  };


  /**
   * Handler for clicking on an existing line.
   */
  const handleLineClick = (e, index) => {
    e.stopPropagation(); 
    if (startPoint) return;

    if (mainRef.current) {
      const rect = mainRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const isNearNode = nodes.some(node => {
        const dx = x - node.x;
        const dy = y - node.y;
        return Math.sqrt(dx * dx + dy * dy) <= SNAP_RADIUS;
      });
      if (isNearNode) return;
    }
    
    setSelectedLineInfo({
      index: index,
      x: e.clientX,
      y: e.clientY
    });
    setEditingValue(lines[index].value);
    setStartPoint(null);
    setCurrentPos(null);
  };

  /**
   * Handler for deleting a line from the popup.
   */
  const handleDeleteLine = () => {
    if (selectedLineInfo === null) return;

    const lineToDelete = lines[selectedLineInfo.index];
    const node1 = { x: lineToDelete.x1, y: lineToDelete.y1 };
    const node2 = { x: lineToDelete.x2, y: lineToDelete.y2 };

    const remainingLines = lines.filter((_, i) => i !== selectedLineInfo.index);
    
    const isNodeConnected = (node, linesList) => {
      return linesList.some(l => 
        (l.x1 === node.x && l.y1 === node.y) || (l.x2 === node.x && l.y2 === node.y)
      );
    };

    const node1Connected = isNodeConnected(node1, remainingLines);
    const node2Connected = isNodeConnected(node2, remainingLines);

    setLines(remainingLines);
    setNodes(prevNodes => prevNodes.filter(n => {
      if (n.x === node1.x && n.y === node1.y) return node1Connected;
      if (n.x === node2.x && n.y === node2.y) return node2Connected;
      return true;
    }));
    setSelectedLineInfo(null);
  };


  // --- Prepare data for export ---
  const graphData = {
    nodes: nodes,
    edges: lines.map(line => ({ 
      from: { x: line.x1, y: line.y1 }, 
      to: { x: line.x2, y: line.y2 },
      component: line.component,
      value: line.value       
    }))
  };
  const graphDataString = JSON.stringify(graphData, null, 2);

  // --- Handle copying to clipboard ---
  const handleCopy = () => {
    if (dataTextRef.current) {
      dataTextRef.current.select();
      try {
        document.execCommand('copy');
        setCopySuccess('Copied to clipboard!');
      } catch (err) {
        setCopySuccess(err + 'Failed to copy.');
      }
      setTimeout(() => setCopySuccess(''), 2000);
      dataTextRef.current.blur();
    }
  };

  // --- Handle loading data from JSON ---
  const handleLoadData = () => {
    setImportError('');
    try {
      const data = JSON.parse(importJson);

      if (!data || !Array.isArray(data.nodes) || !Array.isArray(data.edges)) {
        setImportError("Invalid data: Must have 'nodes' and 'edges' arrays.");
        return;
      }

      const defaultComponent = componentOptions[0].id; // 'wire'
      const loadedLines = data.edges.map(edge => {
        const componentId = edge.component || defaultComponent;
        // Wires have val 2, others default to 10
        const defaultValue = (componentId === 'wire' ? 2 : 10); 
        
        return {
          x1: edge.from.x,
          y1: edge.from.y,
          x2: edge.to.x,
          y2: edge.to.y,
          component: componentId,
          value: edge.value || defaultValue
        };
      });

      setNodes(data.nodes);
      setLines(loadedLines);
      setShowImportPanel(false);
      setImportJson('');

    } catch (err) {
      setImportError(err + 'Invalid JSON format. Please check the text.');
    }
  };

  // --- Handle saving the new value from the popup ---
  const handleValueSave = () => {
    if (!selectedLineInfo) return;
    
    const lineIndex = selectedLineInfo.index;
    const componentId = lines[lineIndex].component;
    
    // Wires have a fixed value and cannot be edited
    if (componentId === 'wire') {
      setEditingValue(2); // Reset local state
      return; 
    }
    
    let newValue = parseInt(editingValue, 10);
    if (isNaN(newValue) || newValue < 1) {
      newValue = 1; // Min value is 1
    }
    
    setLines(prevLines => prevLines.map((line, i) => 
      i === lineIndex ? { ...line, value: newValue } : line
    ));
    setEditingValue(newValue); 
  };

  // --- Handle Enter key in value input ---
  const handleValueKeydown = (e) => {
    if (e.key === 'Enter') {
      handleValueSave();
      e.target.blur();
    }
  };


  return (
    <>     
      <div 
        ref={mainRef}
        className='main'
        onClick={handleClick}
        onMouseMove={handleMouseMove}
      >
        {/* --- Top UI Controls --- */}
        <div className="top-controls">
          <button
            className="ui-button"
            onClick={() => {
              setShowImportPanel(!showImportPanel);
              setShowComponentPanel(false);
              setShowData(false);
            }}
          >
            Import Data
          </button>
          <button
            className="ui-button"
            onClick={() => {
              setShowComponentPanel(!showComponentPanel);
              setShowData(false);
              setShowImportPanel(false);
            }}
          >
            Component
          </button>
          <button
            className="ui-button"
            onClick={() => {
              setShowData(!showData);
              setShowComponentPanel(false);
              setShowImportPanel(false);
            }}
          >
            {showData ? 'Close Data' : 'Export Data'}
          </button>
        </div>

        {/* --- Component Panel --- */}
        {showComponentPanel && (
          <div className="panel component-panel">
            <h3>Select Component</h3>
            <div className="component-swatches">
              {componentOptions.map((component) => (
                <button
                  key={component.id}
                  className="component-swatch"
                  onClick={() => setSelectedComponentId(component.id)}
                >
                  <div 
                    className="component-swatch-color"
                    style={{ backgroundColor: component.color }}
                    data-selected={selectedComponentId === component.id}
                  ></div>
                  <span className="component-swatch-name">{component.name}</span>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* --- Export Panel --- */}
        {showData && (
          <div className="panel export-panel">
            <h3>Graph Data (Nodes & Edges)</h3>
            <div>
              <button className="copy-button" onClick={handleCopy}>Copy JSON</button>
              {copySuccess && <span className="copy-success">{copySuccess}</span>}
            </div>
            <pre>{graphDataString}</pre>
          </div>
        )}
        
        {/* --- Import Panel --- */}
        {showImportPanel && (
          <div className="panel import-panel">
            <h3>Import Data</h3>
            <p style={{ fontSize: '0.875rem', color: '#4b5563', margin: '0 0 0.5rem 0' }}>
              Paste your JSON data below and click "Load".
            </p>
            <textarea
              value={importJson}
              onChange={(e) => {
                setImportJson(e.target.value);
                setImportError('');
              }}
              placeholder='{ "nodes": [...], "edges": [...] }'
            />
            <button className="load-button" onClick={handleLoadData}>
              Load Data
            </button>
            {importError && (
              <p className="import-error">{importError}</p>
            )}
          </div>
        )}

        {/* --- Line Info Popup --- */}
        {selectedLineInfo && (() => {
          const selectedComponent = getComponentById(lines[selectedLineInfo.index].component);
          const isWire = selectedComponent.id === 'wire';

          return (
            <div 
              className="line-popup" 
              style={{ 
                top: `${selectedLineInfo.y + 10}px`, 
                left: `${selectedLineInfo.x + 10}px` 
              }}
              onClick={(e) => e.stopPropagation()}
            >
              <div className="line-popup-header">
                <div 
                  className="line-popup-color-swatch" 
                  style={{ backgroundColor: selectedComponent.color }}
                ></div>
                <span className="line-popup-component-name">
                  {selectedComponent.name}
                </span>
              </div>
              
              <div className="line-popup-value">
                <label htmlFor="value-input">Value:</label>
                <input
                  id="value-input"
                  type="number"
                  value={editingValue}
                  onChange={(e) => setEditingValue(e.target.value)}
                  onBlur={handleValueSave}
                  onKeyDown={handleValueKeydown}
                  min="1"
                  disabled={isWire}
                />
              </div>
              
              <button className="delete-button" onClick={handleDeleteLine}>
                Delete Component
              </button>
            </div>
          );
        })()}


        {/* --- Hidden textarea for copy/paste --- */}
        <textarea
          ref={dataTextRef}
          value={graphDataString}
          readOnly
          className="hidden-textarea"
        />

        {/* --- SVG overlay for drawing --- */}
        <svg
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
          }}
        >

          {/* --- Render all the nodes --- */}
          {nodes.map((node, index) => (
            <circle
              key={`node-${index}`}
              cx={node.x}
              cy={node.y}
              r="5"
              fill="rgb(31 41 55)"
              stroke="rgba(0,0,0,0.3)"
              strokeWidth="1"
              style={{ pointerEvents: 'none' }}
            />
          ))}

          {/* --- Render all the completed lines (components) --- */}
          {lines.map((line, index) => {
            const component = getComponentById(line.component);
            
            return (
              <g 
                key={`line-group-${index}`} 
                onClick={(e) => handleLineClick(e, index)}
                style={{ cursor: 'pointer' }}
              >
                {/* Render the actual component visual */}
                <CircuitComponent 
                  x1={line.x1} y1={line.y1}
                  x2={line.x2} y2={line.y2}
                  component={component}
                  value={line.value}
                />
                
                {/* Invisible hitbox (always 10px) */}
                <line
                  x1={line.x1} y1={line.y1}
                  x2={line.x2} y2={line.y2}
                  stroke="transparent"
                  strokeWidth="10" 
                  style={{ pointerEvents: 'stroke' }}
                />
              </g>
            );
          })}
          
          {/* --- Render the dashed preview line --- */}
          {startPoint && currentPos && (
            <line
              x1={startPoint.x}
              y1={startPoint.y}
              x2={currentPos.x}
              y2={currentPos.y}
              stroke="rgb(107 114 128)"
              strokeWidth="2"
              strokeDasharray="4 4"
              style={{ pointerEvents: 'none' }}
            />
          )}
        </svg>

        <div>
          <h1>Circuit Builder</h1>
        </div>
      </div>
    </>
  );
}

export default App;


