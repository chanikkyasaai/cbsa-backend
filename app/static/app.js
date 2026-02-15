const statusEl = document.getElementById("ws-status");
const deviceInfoEl = document.getElementById("device-info-content");
const eventListEl = document.getElementById("event-list");
const graphPanel = document.querySelector(".graph-panel");

const state = {
  nodes: [],
  links: [],
  nodeById: new Map(),
  linkByKey: new Map(),
  lastEventId: null,
  simulation: null,
  zoom: null,
  svg: null,
  g: null,
  nodeSelection: null,
  linkSelection: null,
};

function setStatus(text, online) {
  statusEl.textContent = text;
  statusEl.classList.toggle("online", online);
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function addEventLog(eventType, receivedAt, mapped) {
  const item = document.createElement("li");
  if (!mapped) {
    item.classList.add("unmapped");
  }
  const time = receivedAt ? new Date(receivedAt * 1000) : new Date();
  const timeLabel = time.toLocaleTimeString();
  item.innerHTML = `${escapeHtml(eventType)} <span>${escapeHtml(timeLabel)}</span>`;
  eventListEl.appendChild(item);

  while (eventListEl.children.length > 200) {
    eventListEl.removeChild(eventListEl.firstChild);
  }

  eventListEl.scrollTop = eventListEl.scrollHeight;
}

function renderDeviceInfo(info) {
  if (!info || typeof info !== "object") {
    deviceInfoEl.textContent = "No device data yet.";
    return;
  }

  deviceInfoEl.innerHTML = "";
  Object.entries(info).forEach(([key, value]) => {
    const row = document.createElement("div");
    const left = document.createElement("span");
    const right = document.createElement("span");
    left.textContent = key;
    right.textContent = String(value);
    row.appendChild(left);
    row.appendChild(right);
    deviceInfoEl.appendChild(row);
  });
}

function buildGraph(mapData) {
  const eventFlowMap = mapData.eventFlowMap || {};
  const nodeMap = new Map();
  const links = [];
  const linkSet = new Set();

  const ensureNode = (id, meta = {}) => {
    if (!id) {
      return;
    }
    if (!nodeMap.has(id)) {
      nodeMap.set(id, {
        id,
        description: meta.description || "",
        pattern: meta.pattern || "",
        group: meta.group || "",
        lastPayload: null,
        active: false,
      });
    }
  };

  const addLink = (source, target) => {
    if (!source || !target) {
      return;
    }
    const key = `${source}-->${target}`;
    if (linkSet.has(key)) {
      return;
    }
    linkSet.add(key);
    links.push({ source, target, id: key, active: false });
  };

  Object.entries(eventFlowMap).forEach(([groupName, events]) => {
    Object.entries(events).forEach(([eventKey, eventInfo]) => {
      ensureNode(eventKey, {
        description: eventInfo.description,
        pattern: eventInfo.pattern,
        group: groupName,
      });

      if (Array.isArray(eventInfo.nextEvents)) {
        eventInfo.nextEvents.forEach((nextEvent) => {
          ensureNode(nextEvent, { group: groupName });
          addLink(eventKey, nextEvent);
        });
      }

      if (eventInfo.specialCase && typeof eventInfo.specialCase === "object") {
        Object.entries(eventInfo.specialCase).forEach(([specialKey, specialInfo]) => {
          ensureNode(specialKey, {
            description: specialInfo.description,
            group: groupName,
          });
          addLink(eventKey, specialKey);

          if (Array.isArray(specialInfo.nextEvents)) {
            specialInfo.nextEvents.forEach((nextEvent) => {
              ensureNode(nextEvent, { group: groupName });
              addLink(specialKey, nextEvent);
            });
          }
        });
      }
    });
  });

  const nodes = Array.from(nodeMap.values());
  state.nodes = nodes;
  state.links = links;
  state.nodeById = nodeMap;
  state.linkByKey = new Map(links.map((link) => [link.id, link]));
}

function setupGraph() {
  const svg = d3.select("#graph");
  const g = svg.append("g").attr("class", "graph-layer");
  state.svg = svg;
  state.g = g;

  const link = g
    .selectAll("line")
    .data(state.links)
    .enter()
    .append("line")
    .attr("class", "link");

  const node = g
    .selectAll("g")
    .data(state.nodes)
    .enter()
    .append("g")
    .attr("class", "node")
    .call(
      d3
        .drag()
        .on("start", (event, d) => {
          if (!event.active) state.simulation.alphaTarget(0.3).restart();
          d.fx = d.x;
          d.fy = d.y;
        })
        .on("drag", (event, d) => {
          d.fx = event.x;
          d.fy = event.y;
        })
        .on("end", (event, d) => {
          if (!event.active) state.simulation.alphaTarget(0);
          d.fx = null;
          d.fy = null;
        })
    );

  node.append("circle").attr("r", 10);
  node.append("text").attr("class", "node-label").text((d) => d.id);

  const tooltip = d3.select("#tooltip");

  node
    .on("mouseenter", (event, d) => {
      const payload = d.lastPayload
        ? JSON.stringify(d.lastPayload, null, 2)
        : "No payload yet.";
      const pattern = d.pattern ? `Pattern: ${d.pattern}` : "";
      tooltip
        .style("display", "block")
        .html(
          `<div class="title">${escapeHtml(d.id)}</div>` +
            `<div class="meta">${escapeHtml(d.description || "")}</div>` +
            `<div class="meta">${escapeHtml(pattern)}</div>` +
            `<pre>${escapeHtml(payload)}</pre>`
        );
    })
    .on("mousemove", (event) => {
      const rect = graphPanel.getBoundingClientRect();
      tooltip
        .style("left", `${event.clientX - rect.left + 20}px`)
        .style("top", `${event.clientY - rect.top + 20}px`);
    })
    .on("mouseleave", () => {
      tooltip.style("display", "none");
    });

  const simulation = d3
    .forceSimulation(state.nodes)
    .force(
      "link",
      d3.forceLink(state.links).id((d) => d.id).distance(110).strength(0.7)
    )
    .force("charge", d3.forceManyBody().strength(-220))
    .force("center", d3.forceCenter(0, 0))
    .force("collide", d3.forceCollide().radius(26));

  simulation.on("tick", () => {
    link
      .attr("x1", (d) => d.source.x)
      .attr("y1", (d) => d.source.y)
      .attr("x2", (d) => d.target.x)
      .attr("y2", (d) => d.target.y);

    node.attr("transform", (d) => `translate(${d.x}, ${d.y})`);
  });

  state.simulation = simulation;
  state.nodeSelection = node;
  state.linkSelection = link;

  const zoom = d3
    .zoom()
    .scaleExtent([0.3, 3])
    .on("zoom", (event) => g.attr("transform", event.transform));

  svg.call(zoom);
  state.zoom = zoom;

  document.getElementById("zoom-in").addEventListener("click", () => {
    svg.transition().call(zoom.scaleBy, 1.2);
  });
  document.getElementById("zoom-out").addEventListener("click", () => {
    svg.transition().call(zoom.scaleBy, 0.8);
  });
  document.getElementById("zoom-reset").addEventListener("click", () => {
    svg.transition().call(zoom.transform, d3.zoomIdentity);
  });

  resizeGraph();
  window.addEventListener("resize", resizeGraph);
}

function resizeGraph() {
  const rect = document.getElementById("graph").getBoundingClientRect();
  const width = rect.width;
  const height = rect.height;
  state.svg.attr("width", width).attr("height", height);
  if (state.simulation) {
    state.simulation.force("center", d3.forceCenter(width / 2, height / 2));
    state.simulation.alpha(0.4).restart();
  }
}

function updateStyles() {
  state.nodeSelection.classed("active", (d) => d.active);
  state.linkSelection.classed("active", (d) => d.active);
}

function handleMonitorMessage(message) {
  const eventType =
    message.eventType ||
    (message.payload && message.payload.eventType) ||
    "UNKNOWN_EVENT";

  if (message.deviceInfo) {
    renderDeviceInfo(message.deviceInfo);
  }

  const node = state.nodeById.get(eventType);
  const mapped = Boolean(node);

  addEventLog(eventType, message.receivedAt, mapped);

  if (node) {
    node.lastPayload = message.payload || message.raw || message;
    node.active = true;

    if (state.lastEventId) {
      const key = `${state.lastEventId}-->${eventType}`;
      const link = state.linkByKey.get(key);
      if (link) {
        link.active = true;
      }
    }

    state.lastEventId = eventType;
    updateStyles();
  }
}

function connectMonitor() {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const url = `${protocol}://${window.location.host}/ws/monitor`;

  const socket = new WebSocket(url);

  socket.onopen = () => {
    setStatus("Live", true);
  };

  socket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      handleMonitorMessage(data);
    } catch (error) {
      console.error("Invalid monitor payload", error);
    }
  };

  socket.onclose = () => {
    setStatus("Reconnecting...", false);
    setTimeout(connectMonitor, 2000);
  };

  socket.onerror = () => {
    setStatus("Connection error", false);
  };
}

async function init() {
  setStatus("Connecting...", false);

  try {
    const response = await fetch("/event-flow-map");
    const mapData = await response.json();
    buildGraph(mapData);
    setupGraph();
  } catch (error) {
    console.error("Failed to load event flow map", error);
  }

  connectMonitor();
}

init();
