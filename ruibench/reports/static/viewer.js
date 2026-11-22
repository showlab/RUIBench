/* RUIBench episode viewer.  Reads the JSON in #ruibench-data and, for each
   episode, draws a plan view replayed from the logged trajectory (any engine),
   the recorded camera frames (Isaac runs made with --record), and a trace of
   model turns, arm actions, milestones and events, all on one clock.
   Plan orientation matches the fixed camera: the robot at the top, +y to the
   right, x growing toward the camera. */
(() => {
  "use strict";
  const DATA = JSON.parse(document.getElementById("ruibench-data").textContent);
  const INITIAL_HASH = location.hash;              // before selecting an episode rewrites it
  const SVGNS = "http://www.w3.org/2000/svg";

  // ------------------------------------------------------------ helpers
  function h(tag, attrs, ...kids) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === null || v === undefined || v === false) continue;
      if (k === "class") node.className = v;
      else if (k === "text") node.textContent = v;
      else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v === true ? "" : v);
    }
    for (const kid of kids.flat()) {
      if (kid === null || kid === undefined || kid === false) continue;
      node.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
    }
    return node;
  }
  // theme tokens go through style: var() in a presentation attribute is not
  // honoured everywhere, in a style declaration it is
  const STYLED = new Set(["fill", "stroke", "font-family", "stop-color"]);
  function s(tag, attrs, ...kids) {
    const node = document.createElementNS(SVGNS, tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === null || v === undefined) continue;
      if (k === "text") node.textContent = v;
      else if (STYLED.has(k) && typeof v === "string" && v.startsWith("var(")) node.style.setProperty(k, v);
      else node.setAttribute(k, v);
    }
    for (const kid of kids.flat()) if (kid) node.append(kid);
    return node;
  }
  const title = (text) => s("title", { text });
  const num = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v)) ? "–" : Number(v).toFixed(d);
  const secs = (v) => `${num(v, 1)} s`;
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const tick = (v) => (Math.abs(v) < 1e-9 ? 0 : v).toFixed(1);

  /** largest i with ts[i] <= t (0 if t precedes everything) */
  function indexAt(ts, t) {
    let lo = 0, hi = ts.length - 1;
    if (hi < 0 || t <= ts[0]) return 0;
    if (t >= ts[hi]) return hi;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (ts[mid] <= t) lo = mid; else hi = mid - 1;
    }
    return lo;
  }
  function sampleRow(rows, ts, t) {
    const i = indexAt(ts, t);
    const a = rows[i], b = rows[Math.min(i + 1, rows.length - 1)];
    const span = ts[Math.min(i + 1, ts.length - 1)] - ts[i];
    const f = span > 0 ? clamp((t - ts[i]) / span, 0, 1) : 0;
    // interpolate the continuous fields; take flags from the earlier sample
    return a.map((v, k) => (k >= 4 || typeof v !== "number") ? v : v + (b[k] - v) * f);
  }
  function describeCommand(c) {
    if (typeof c !== "object" || c === null) return String(c);
    const arm = c.arm ? ` ${c.arm}` : "";
    switch (c.op) {
      case "move_to": return `move_to${arm} → (${(c.pos || []).map((v) => num(v)).join(", ")})`;
      case "move_delta": return `move_delta${arm} by (${(c.delta || c.dpos || []).map((v) => num(v)).join(", ")})`;
      case "gripper": return `gripper${arm} ${c.value >= 0.5 ? "close" : "open"} (${c.value})`;
      case "wait": return `wait ${c.seconds} s`;
      case "done": return `done${c.note ? ` — ${c.note}` : ""}`;
      case "cancel": return `cancel${arm}`;
      default: return JSON.stringify(c);
    }
  }
  function niceStep(span, target) {
    const raw = span / target;
    const p = Math.pow(10, Math.floor(Math.log10(raw)));
    for (const m of [1, 2, 5, 10]) if (m * p >= raw) return m * p;
    return 10 * p;
  }
  function outcomePill(ep) {
    const o = ep.outcome;
    if (o.coverageGap) return h("span", { class: "pill warn", text: "coverage gap" });
    if (o.failure === "harness_error") return h("span", { class: "pill warn", text: "harness error" });
    return o.success
      ? h("span", { class: "pill ok" }, "✓ success")
      : h("span", { class: "pill fail" }, `✗ ${o.failure.replace(/_/g, " ")}`);
  }

  // ------------------------------------------------------------ plan view
  const PLAN_W = 560, PAD_L = 40, PAD_T = 26, PAD_R = 10, PAD_B = 12;

  function planGeometry(scene, extraPoints) {
    const xs = [], ys = [];
    const add = (x, y, r = 0) => { xs.push(x - r, x + r); ys.push(y - r, y + r); };
    for (const r of scene.robots) add(r.base[0], r.base[1], 0.09);
    for (const b of scene.bodies) add(b.pos[0], b.pos[1], Math.hypot(b.size[0], b.size[1]));
    for (const m of scene.markers || []) add(m.pos[0], m.pos[1], Math.max(m.size[0], m.size[1]));
    for (const p of extraPoints || []) add(p[0], p[1], 0.02);
    const step = 0.05;
    const xMin = Math.floor((Math.min(...xs) - 0.03) / step) * step;
    const xMax = Math.ceil((Math.max(...xs) + 0.03) / step) * step;
    const yMin = Math.floor((Math.min(...ys) - 0.03) / step) * step;
    const yMax = Math.ceil((Math.max(...ys) + 0.03) / step) * step;
    const scale = (PLAN_W - PAD_L - PAD_R) / (yMax - yMin);
    const height = PAD_T + (xMax - xMin) * scale + PAD_B;
    const map = (x, y) => [PAD_L + (y - yMin) * scale, PAD_T + (x - xMin) * scale];
    return { xMin, xMax, yMin, yMax, scale, width: PLAN_W, height, map };
  }

  function footprint(geo, pos, yaw, hx, hy) {
    const c = Math.cos(yaw), sn = Math.sin(yaw);
    return [[hx, hy], [hx, -hy], [-hx, -hy], [-hx, hy]]
      .map(([a, b]) => geo.map(pos[0] + c * a - sn * b, pos[1] + sn * a + c * b))
      .map((p) => p.map((v) => v.toFixed(1)).join(",")).join(" ");
  }

  function drawBody(geo, b, pos, yaw, opts) {
    const round = b.shape === "cylinder" || b.shape === "sphere" || b.footprint === "round";
    const fill = b.color || "#9aa3ae";
    const g = s("g", {}, title(`${b.name}${b.category ? ` · ${b.category}` : ""}`));
    if (b.container) {
      const wall = Math.max(2.5, (b.wall || 0.01) * geo.scale);
      g.append(s("polygon", { points: footprint(geo, pos, yaw, b.size[0], b.size[1]),
        fill, "fill-opacity": 0.16, stroke: fill, "stroke-width": wall, "stroke-linejoin": "round" }));
      return g;
    }
    const attrs = {
      fill, "fill-opacity": b.fixed ? 0.55 : 1,
      stroke: "var(--plan-ink)", "stroke-opacity": b.fixed ? 0.35 : 0.7, "stroke-width": 1,
    };
    if (round) {
      const [u, v] = geo.map(pos[0], pos[1]);
      g.append(s("circle", { cx: u.toFixed(1), cy: v.toFixed(1), r: (b.size[0] * geo.scale).toFixed(1), ...attrs }));
    } else {
      g.append(s("polygon", { points: footprint(geo, pos, yaw, b.size[0], b.size[1]), ...attrs }));
    }
    if (opts && opts.held) {
      const [u, v] = geo.map(pos[0], pos[1]);
      const r = Math.hypot(b.size[0], b.size[1]) * geo.scale + 5;
      g.append(s("circle", { cx: u.toFixed(1), cy: v.toFixed(1), r: r.toFixed(1), fill: "none",
        stroke: "var(--ink)", "stroke-width": 1.4, "stroke-dasharray": "3 3" }));
    }
    return g;
  }

  function drawMarker(geo, m) {
    const [u, v] = geo.map(m.pos[0], m.pos[1]);
    const r = Math.max(m.size[0], m.size[1]) * geo.scale;
    const stroke = m.color && m.color.toLowerCase() !== "#f2f2f2" ? m.color : "var(--plan-ink)";
    const g = s("g", { stroke, "stroke-width": 1.6, fill: "none" }, title(`${m.name} · marker`));
    if (m.strokes && m.strokes.length) {
      // the same strokes the renderer draws
      const width = Math.max(1.2, (m.width || 0.006) * geo.scale);
      g.setAttribute("stroke-width", width.toFixed(1));
      g.setAttribute("stroke-linecap", "square");
      for (const [x1, y1, x2, y2] of m.strokes) {
        const [a, b] = geo.map(x1, y1), [c, d] = geo.map(x2, y2);
        g.append(s("line", { x1: a.toFixed(1), y1: b.toFixed(1), x2: c.toFixed(1), y2: d.toFixed(1) }));
      }
      return g;
    }
    if (m.shape === "cross") {
      g.append(s("line", { x1: u - r, y1: v - r, x2: u + r, y2: v + r }));
      g.append(s("line", { x1: u - r, y1: v + r, x2: u + r, y2: v - r }));
    } else {
      g.append(s("circle", { cx: u, cy: v, r, "stroke-dasharray": "3 2" }));
    }
    return g;
  }

  function drawPlanFrame(svg, geo, scene) {
    svg.setAttribute("viewBox", `0 0 ${geo.width} ${geo.height.toFixed(0)}`);
    svg.append(s("rect", { x: PAD_L, y: PAD_T, width: geo.width - PAD_L - PAD_R,
      height: geo.height - PAD_T - PAD_B, fill: "var(--plan-table)", rx: 4 }));
    const grid = s("g", { "shape-rendering": "crispEdges" });
    const labels = s("g", { fill: "var(--muted)", "font-size": 10, "font-family": "var(--font-mono)" });
    for (let x = geo.xMin; x <= geo.xMax + 1e-9; x += 0.05) {
      const major = Math.abs(Math.round(x * 10) - x * 10) < 1e-6;
      const [, v] = geo.map(x, geo.yMin);
      grid.append(s("line", { x1: PAD_L, x2: geo.width - PAD_R, y1: v, y2: v,
        stroke: major ? "var(--plan-grid-major)" : "var(--plan-grid)", "stroke-width": major ? 1 : 0.6 }));
      if (major) labels.append(s("text", { x: PAD_L - 5, y: v + 3, "text-anchor": "end", text: tick(x) }));
    }
    for (let y = geo.yMin; y <= geo.yMax + 1e-9; y += 0.05) {
      const major = Math.abs(Math.round(y * 10) - y * 10) < 1e-6;
      const [u] = geo.map(geo.xMin, y);
      grid.append(s("line", { y1: PAD_T, y2: geo.height - PAD_B, x1: u, x2: u,
        stroke: major ? "var(--plan-grid-major)" : "var(--plan-grid)", "stroke-width": major ? 1 : 0.6 }));
      if (major) labels.append(s("text", { x: u, y: PAD_T - 6, "text-anchor": "middle", text: tick(y) }));
    }
    labels.append(s("text", { x: 2, y: 9, "text-anchor": "start", text: "metres · x ↓ y →" }));
    svg.append(grid, labels);
    for (const r of scene.robots) {
      const [u, v] = geo.map(r.base[0], r.base[1]);
      svg.append(s("g", {}, title(`${r.name} base`),
        s("circle", { cx: u, cy: v, r: 0.075 * geo.scale, fill: "var(--surface)",
          stroke: "var(--plan-ink)", "stroke-width": 1.2 }),
        s("text", { x: u, y: v + 3.5, "text-anchor": "middle", "font-size": 10,
          "font-family": "var(--font-label)", fill: "var(--plan-ink)", text: r.name })));
    }
    for (const m of scene.markers || []) svg.append(drawMarker(geo, m));
    const fixed = s("g", {});
    for (const b of scene.bodies) if (b.fixed && !b.attachedTo) fixed.append(drawBody(geo, b, b.pos, b.yaw));
    svg.append(fixed);
    const live = s("g", {});
    svg.append(live);
    return live;
  }

  function drawArm(geo, tableH, id, row, trail, label) {
    const [x, y, z, grip] = row;
    const [u, v] = geo.map(x, y);
    const g = s("g", {}, title(`${id} · tool at (${num(x)}, ${num(y)}, ${num(z)}) · gripper ${grip >= 0.5 ? "closed" : "open"}`));
    if (trail.length > 1) {
      g.append(s("polyline", { points: trail.map((p) => geo.map(p[0], p[1]).map((q) => q.toFixed(1)).join(",")).join(" "),
        fill: "none", stroke: "var(--ink)", "stroke-opacity": 0.35, "stroke-width": 1.5,
        "stroke-linecap": "round", "stroke-linejoin": "round" }));
    }
    const lift = clamp((z - tableH) * 55, 0, 26);
    g.append(s("circle", { cx: u, cy: v, r: (5 + lift).toFixed(1), fill: "none", stroke: "var(--ink)",
      "stroke-width": 1.2, "stroke-opacity": 0.8 }));
    const gap = (0.04 * (1 - clamp(grip, 0, 1)) + 0.006) * geo.scale;
    const len = 0.016 * geo.scale;
    for (const side of [-1, 1]) {
      g.append(s("rect", { x: (u + side * gap - 1.6).toFixed(1), y: (v - len).toFixed(1), width: 3.2,
        height: (2 * len).toFixed(1), rx: 1, fill: "var(--ink)" }));
    }
    g.append(s("circle", { cx: u, cy: v, r: 1.6, fill: "var(--ink)" }));
    if (label) g.append(s("text", { x: u + 8 + lift, y: v - 6 - lift / 2, "font-size": 10,
      "font-family": "var(--font-label)", fill: "var(--ink-2)", text: id }));
    return g;
  }

  function makePlan(ep) {
    const traj = ep.traj;
    const armPoints = Object.values(traj.arms).flat().filter((_, i) => i % 5 === 0);
    const geo = planGeometry(ep.scene, armPoints);
    const svg = s("svg", { role: "img", "aria-label": `Plan view of ${ep.task}` });
    const live = drawPlanFrame(svg, geo, ep.scene);
    const bodies = Object.fromEntries(ep.scene.bodies.map((b) => [b.name, b]));
    const armIds = Object.keys(traj.arms);
    const tableH = ep.scene.tableHeight;
    function setTime(t) {
      live.replaceChildren();
      const i = indexAt(traj.t, t);
      for (const [name, rows] of Object.entries(traj.objects)) {
        const b = bodies[name];
        if (!b || (b.fixed && !b.attachedTo)) continue;
        const row = sampleRow(rows, traj.t, t);
        // a cloth's row also carries the box round it at that moment
        const now = row.length > 6 ? { ...b, size: [row[5], row[6], b.size[2]] } : b;
        live.append(drawBody(geo, now, row, row[3], { held: rows[i][4] > 0 }));
      }
      for (const id of armIds) {
        const rows = traj.arms[id];
        const start = indexAt(traj.t, t - 3);
        const trail = rows.slice(start, i + 1);
        live.append(drawArm(geo, tableH, id, sampleRow(rows, traj.t, t), trail, armIds.length > 1));
      }
    }
    return { svg, setTime };
  }

  function planLegend(scene) {
    const seen = new Map();
    for (const b of scene.bodies) {
      const kind = b.container ? "bin" : (b.fixed ? "fixed" : "loose");
      const key = `${b.color}|${kind}`;
      if (!seen.has(key)) seen.set(key, { color: b.color, kind, names: [] });
      seen.get(key).names.push(b.name);
    }
    const items = [...seen.values()].map((e) => h("span", { title: e.names.join(", ") },
      h("i", { style: `background:${e.color};opacity:${e.kind === "loose" ? 1 : 0.55}` }),
      e.kind === "bin" ? `bin (${e.names.length})` : `${e.kind} ×${e.names.length}`));
    items.push(h("span", {}, "○ tool — ring grows with height, bars = fingers, dashed = held"));
    return h("div", { class: "legend" }, items);
  }

  // ------------------------------------------------------------ cameras
  function episodeVideo(ep, label) {
    if (!ep.video) return null;
    return h("figure", { class: "clip" },
      h("video", { src: ep.video, controls: true, muted: true, playsinline: true, preload: "metadata" }),
      h("figcaption", { class: "meta", text: label || "the whole episode · fixed camera | wrist · 2× speed" }));
  }

  function makeCameras(ep) {
    const names = Object.keys(ep.frames || {});
    if (!names.length && ep.video) {
      return { node: episodeVideo(ep), setTime() {} };
    }
    if (!names.length) {
      const engineNote = ep.engine === "isaac"
        ? h("span", {}, "This run was not recorded. Re-run it with ", h("code", { text: "ruibench run … --engine isaac --record" }), ".")
        : h("span", {}, `The ${ep.engine} engine renders no pixels; the plan view is the replay. Camera frames come from `, h("code", { text: "--engine isaac --record" }), ".");
      return { node: h("div", { class: "empty" }, h("b", { text: "No camera frames" }), engineNote), setTime() {} };
    }
    const cards = names.map((name) => {
      const img = h("img", { alt: `${name} camera`, loading: "lazy" });
      const when = h("div", { class: "when" });
      const frames = ep.frames[name];
      const ts = frames.map((f) => f[0]);
      let shown = -1;
      return {
        node: h("div", { class: "cam" }, h("div", { class: "label", text: name }),
          h("div", { class: "frame" }, img), when),
        setTime(t) {
          const i = indexAt(ts, t);
          if (i === shown) return;
          shown = i;
          img.src = frames[i][1];
          when.textContent = `${secs(frames[i][0])} · ${frames[i][2] === "observation" ? "shown to the model" : "between turns"}`;
        },
      };
    });
    return {
      node: h("div", { class: "cam-stack" }, h("div", { class: "cams" }, cards.map((c) => c.node)), episodeVideo(ep)),
      setTime(t) { for (const c of cards) c.setTime(t); },
    };
  }

  // ------------------------------------------------------------ trace
  const TR_W = 1000, TR_LABEL = 118, TR_LANE = 24, TR_GAP = 7, TR_TOP = 28, TR_AXIS = 24;

  function makeTrace(ep, onSeek) {
    const T = Math.max(ep.duration, 0.1);
    const arms = ep.scene.armIds.length ? ep.scene.armIds : Object.keys(ep.traj.arms);
    const lanes = [{ key: "model", label: "model turns" },
      ...arms.map((a) => ({ key: a, label: a })), { key: "marks", label: "milestones" }];
    const height = TR_TOP + lanes.length * (TR_LANE + TR_GAP) + TR_AXIS;
    const x = (t) => TR_LABEL + (clamp(t, 0, T) / T) * (TR_W - TR_LABEL - 12);
    const laneY = (k) => TR_TOP + lanes.findIndex((l) => l.key === k) * (TR_LANE + TR_GAP);
    const svg = s("svg", { viewBox: `0 0 ${TR_W} ${height}`, role: "img",
      "aria-label": "Timeline of model turns, arm actions and milestones" });

    for (const lane of lanes) {
      const y = laneY(lane.key);
      svg.append(s("rect", { x: TR_LABEL, y, width: TR_W - TR_LABEL - 12, height: TR_LANE, fill: "var(--lane)", rx: 3 }));
      svg.append(s("text", { x: TR_LABEL - 8, y: y + TR_LANE / 2 + 4, "text-anchor": "end",
        "font-size": 11, "font-family": "var(--font-label)", fill: "var(--muted)", text: lane.label }));
    }
    // time axis
    const step = niceStep(T, 10);
    const axisY = TR_TOP + lanes.length * (TR_LANE + TR_GAP) + 4;
    for (let t = 0; t <= T + 1e-9; t += step) {
      svg.append(s("line", { x1: x(t), x2: x(t), y1: TR_TOP - 4, y2: axisY, stroke: "var(--rule)", "stroke-width": 1 }));
      svg.append(s("text", { x: x(t), y: axisY + 13, "text-anchor": "middle", "font-size": 10,
        "font-family": "var(--font-mono)", fill: "var(--muted)", text: `${+t.toFixed(1)}s` }));
    }
    // model turns: from the instant the picture was taken to the instant the answer landed
    const my = laneY("model");
    for (const turn of ep.turns) {
      const x0 = x(turn.obs), x1 = Math.max(x(turn.ret), x0 + 2);
      svg.append(s("rect", { x: x0, y: my + 4, width: x1 - x0, height: TR_LANE - 8, rx: 2,
        fill: "var(--ink-2)", "fill-opacity": 0.55 },
        title(`turn ${turn.i}: saw the world at ${secs(turn.obs)}, answer landed at ${secs(turn.ret)} (observation age ${secs(turn.age)})`)));
    }
    // actions, one lane per arm
    for (const a of ep.actions) {
      if (!lanes.some((l) => l.key === a.arm)) continue;
      const y = laneY(a.arm);
      const t1 = a.t1 === null ? T : a.t1;
      const x0 = x(a.t0), w = Math.max(x(t1) - x0, 2);
      const color = a.status === "failed" ? "var(--fail)"
        : a.kind === "SetGripper" ? "var(--act-grip)"
          : (a.kind === "MoveTo" || a.kind === "MoveDelta") ? "var(--act-move)" : "var(--act-other)";
      const cancelled = a.status === "cancelled";
      svg.append(s("rect", { x: x0, y: y + 3, width: w, height: TR_LANE - 6, rx: 2,
        fill: cancelled ? "none" : color, stroke: cancelled ? color : "var(--surface)",
        "stroke-dasharray": cancelled ? "3 2" : null, "stroke-width": cancelled ? 1.2 : 0.8 },
        title(`${a.kind} · ${a.status} · ${secs(a.t0)} → ${a.t1 === null ? "end" : secs(a.t1)}${a.detail ? ` · ${a.detail}` : ""}`)));
    }
    // milestones and feedback
    const ky = laneY("marks") + TR_LANE / 2;
    const names = Object.entries(ep.outcome.milestones).sort((p, q) => p[1] - q[1]);
    names.forEach(([name, t], k) => {
      const cx = x(t);
      svg.append(s("path", { d: `M${cx} ${ky - 6} L${cx + 6} ${ky} L${cx} ${ky + 6} L${cx - 6} ${ky} Z`,
        fill: "var(--ok)" }, title(`${name.replace(/_/g, " ")} at ${secs(t)}`)));
      const flip = cx > TR_W - 130;
      svg.append(s("text", { x: flip ? cx - 8 : cx + 8, y: ky + (k % 2 ? 9 : -3), "font-size": 9.5,
        "text-anchor": flip ? "end" : "start",
        "font-family": "var(--font-label)", fill: "var(--ink-2)", text: name.replace(/_/g, " ") }));
    });
    for (const f of ep.feedback) {
      svg.append(s("circle", { cx: x(f.t), cy: ky, r: 3.4,
        fill: f.signal === "correct" ? "var(--ok)" : "var(--fail)" },
        title(`feedback for ${f.obj}: ${f.signal} at ${secs(f.t)}`)));
    }
    // events cut across every lane
    for (const e of ep.events) {
      const ex = x(e.t);
      svg.append(s("line", { x1: ex, x2: ex, y1: TR_TOP - 10, y2: axisY, stroke: "var(--warn)",
        "stroke-width": 1.4, "stroke-dasharray": "4 3" }, title(`${e.name} at ${secs(e.t)}: ${e.detail}`)));
      svg.append(s("text", { x: ex + 4, y: TR_TOP - 12, "font-size": 10, "font-family": "var(--font-label)",
        fill: "var(--warn)", text: e.name.replace(/_/g, " ") }));
    }
    // playhead
    const head = s("g", {},
      s("line", { x1: 0, x2: 0, y1: TR_TOP - 16, y2: axisY, stroke: "var(--accent)", "stroke-width": 2 }),
      s("path", { d: "M-5 -2 L5 -2 L0 5 Z", transform: `translate(0 ${TR_TOP - 16})`, fill: "var(--accent)" }));
    svg.append(head);

    function seekFrom(evt) {
      const box = svg.getBoundingClientRect();
      const vx = ((evt.clientX - box.left) / box.width) * TR_W;
      onSeek(clamp((vx - TR_LABEL) / (TR_W - TR_LABEL - 12), 0, 1) * T);
    }
    let dragging = false;
    svg.addEventListener("pointerdown", (e) => { dragging = true; svg.setPointerCapture(e.pointerId); seekFrom(e); });
    svg.addEventListener("pointermove", (e) => { if (dragging) seekFrom(e); });
    svg.addEventListener("pointerup", () => { dragging = false; });
    return { svg, setTime(t) { head.setAttribute("transform", `translate(${x(t).toFixed(1)} 0)`); } };
  }

  // ------------------------------------------------------------ details
  function makeDetails(ep, onSeek) {
    const nowBox = h("div", { class: "panel" });
    const turnsBox = h("div", { class: "turns" });
    const turnNodes = ep.turns.map((turn) => {
      const cmds = Array.isArray(turn.cmds) ? turn.cmds : [turn.cmds];
      const node = h("details", {},
        h("summary", { onclick: () => onSeek(turn.obs) },
          h("span", { class: "label", text: `turn ${turn.i}` }),
          h("span", { class: "mono", text: `saw ${secs(turn.obs)} · answered ${secs(turn.ret)}` }),
          h("span", { class: "meta", text: `${cmds.length} command${cmds.length === 1 ? "" : "s"}` })),
        h("ol", { class: "cmds" }, cmds.map((c) => h("li", { text: describeCommand(c) }))));
      turnsBox.append(node);
      return node;
    });
    const armIds = Object.keys(ep.traj.arms);
    const bodies = Object.keys(ep.traj.objects).filter((n) => {
      const b = ep.scene.bodies.find((q) => q.name === n);
      return b && !b.fixed;
    });
    let lastTurn = -2;
    function setTime(t) {
      const i = indexAt(ep.traj.t, t);
      const armRows = armIds.map((id) => {
        const row = sampleRow(ep.traj.arms[id], ep.traj.t, t);
        const holding = bodies.filter((n) => ep.traj.objects[n][i][4] === armIds.indexOf(id) + 1);
        const active = ep.actions.filter((a) => a.arm === id && a.t0 <= t && (a.t1 === null || t < a.t1));
        return h("div", { class: "arm-now" },
          h("div", { class: "label", text: id }),
          h("dl", { class: "kv" },
            h("dt", { text: "tool" }), h("dd", { class: "mono", text: `(${num(row[0])}, ${num(row[1])}, ${num(row[2])}) m` }),
            h("dt", { text: "gripper" }), h("dd", { text: row[3] >= 0.5 ? "closed" : "open" }),
            h("dt", { text: "holding" }), h("dd", { text: holding.join(", ") || "nothing" }),
            h("dt", { text: "doing" }), h("dd", { text: active.map((a) => `${a.kind} since ${secs(a.t0)}`).join(" · ") || "idle" })));
      });
      const rows = bodies.map((n) => {
        const r = ep.traj.objects[n][i];
        const holder = r[4] > 0 ? armIds[r[4] - 1] : "";
        return h("tr", {}, h("td", { text: n }), h("td", { class: "num", text: num(r[0]) }),
          h("td", { class: "num", text: num(r[1]) }), h("td", { class: "num", text: num(r[2]) }),
          h("td", { text: holder }));
      });
      const turnIndex = ep.turns.reduce((k, turn, j) => (turn.obs <= t + 1e-9 ? j : k), -1);
      if (turnIndex !== lastTurn) {
        turnNodes.forEach((node, j) => node.classList.toggle("now", j === turnIndex));
        lastTurn = turnIndex;
      }
      const turn = ep.turns[turnIndex];
      nowBox.replaceChildren(
        h("div", { class: "panel-head" }, h("span", { class: "label", text: "at this moment" }),
          h("span", { class: "clock" }, h("b", { text: secs(t) }))),
        ...armRows,
        turn ? h("div", {}, h("div", { class: "label", text: `latest turn · ${turn.i}` }),
          h("ol", { class: "cmds" }, (Array.isArray(turn.cmds) ? turn.cmds : [turn.cmds]).map((c) => h("li", { text: describeCommand(c) }))))
          : h("div", { class: "label", text: "no turn yet" }),
        bodies.length ? h("div", { class: "tablewrap" }, h("table", {},
          h("thead", {}, h("tr", {}, ["object", "x", "y", "z", "held by"].map((c) => h("th", { text: c })))),
          h("tbody", {}, rows))) : null);
    }
    const node = h("div", { class: "details" }, nowBox,
      h("div", { class: "panel" }, h("div", { class: "panel-head" },
        h("span", { class: "label", text: "model turns" }),
        h("span", { class: "meta", text: `${ep.turns.length} · click one to jump to it` })), turnsBox));
    return { node, setTime };
  }

  // ------------------------------------------------------------ episode
  let stopPlayback = () => {};

  function icon(kind) {
    const paths = {
      play: "M4 2.5 L13 8 L4 13.5 Z",
      pause: "M4 2.5 H7 V13.5 H4 Z M9 2.5 H12 V13.5 H9 Z",
      restart: "M3 2.5 H5 V13.5 H3 Z M13 2.5 L6 8 L13 13.5 Z",
    };
    return s("svg", { viewBox: "0 0 16 16", width: 14, height: 14, "aria-hidden": "true" },
      s("path", { d: paths[kind], fill: "currentColor" }));
  }

  function metricChips(ep) {
    const chips = [];
    const add = (label, value) => chips.push(h("div", { class: "chip" }, h("span", { class: "label", text: label }), h("span", { class: "v", text: value })));
    const o = ep.outcome;
    const hit = Object.keys(o.milestones).length;
    add("progress", `${Math.round(o.progress * 100)}% · ${hit}/${o.milestoneCount} milestones`);
    add("model calls", String(ep.metrics.model_calls !== undefined ? ep.metrics.model_calls : ep.turns.length));
    add("sim time", secs(ep.metrics.sim_seconds !== undefined ? ep.metrics.sim_seconds : ep.duration));
    if (ep.metrics.wall_seconds !== undefined) add("wall time", secs(ep.metrics.wall_seconds));
    if (ep.metrics.observation_age_mean) add("mean obs. age", secs(ep.metrics.observation_age_mean));
    for (const [k, v] of Object.entries(ep.taskMetrics || {})) {
      if (v && typeof v === "object") {
        for (const [k2, v2] of Object.entries(v)) add(`${k} · ${k2.replace(/_/g, " ")}`, String(v2));
      } else if (typeof v === "number" && k.endsWith("_m")) {
        add(k.replace(/_m$/, "").replace(/_/g, " "), `${(v * 1000).toFixed(1)} mm`);
      } else add(k.replace(/_/g, " "), String(v));
    }
    return h("div", { class: "chips" }, chips);
  }

  function showEpisode(ep, main) {
    stopPlayback();
    let t = 0, playing = false, speed = 1, last = 0, frame = 0;
    const plan = makePlan(ep);
    const cams = makeCameras(ep);
    const trace = makeTrace(ep, (v) => { pause(); setTime(v); });
    const details = makeDetails(ep, (v) => { pause(); setTime(v); });

    const later = ep.events.filter((e) => e.newInstruction).map((e) => e.newInstruction);
    const head = h("header", { class: "head" },
      h("div", { class: "eyebrow" },
        h("span", { class: "pill", text: ep.task }),
        h("span", { class: "label", text: `${ep.suiteName} · ${ep.name}` }),
        h("span", { class: "pill", text: `${ep.level}${ep.knob ? ` · ${ep.knob.field} = ${ep.knob.value}` : ""}` }),
        h("span", { class: "meta mono", text: `seed ${ep.seed} · ${ep.interface} · ${ep.agent}${ep.condition !== "main" ? ` · ${ep.condition}` : ""}` }),
        outcomePill(ep)),
      h("h1", { text: ep.instruction || "(no instruction recorded)" }),
      later.length ? h("div", { class: "quote", text: `changed mid-episode to: “${later.join("”, then “")}”` }) : null,
      ep.resultsGrade ? null : h("div", { class: "notice" }, h("b", { text: `${ep.engine} engine` }),
        "Not a result: only the photorealistic engine produces leaderboard numbers."),
      metricChips(ep));

    const slider = h("input", { type: "range", id: "time", min: 0, max: ep.duration, step: 0.05, value: 0,
      "aria-label": "Episode time", oninput: (e) => { pause(); setTime(Number(e.target.value)); } });
    const clock = h("span", { class: "clock" });
    const playBtn = h("button", { type: "button", id: "play", "aria-label": "Play", onclick: () => (playing ? pause() : play()) }, icon("play"));
    const speedSel = h("select", { id: "speed", "aria-label": "Playback speed", onchange: (e) => { speed = Number(e.target.value); } },
      [0.5, 1, 2, 4, 8].map((v) => h("option", { value: v, selected: v === 1, text: `${v}×` })));
    const transport = h("div", { class: "transport" }, playBtn,
      h("button", { type: "button", "aria-label": "Back to the start", onclick: () => { pause(); setTime(0); } }, icon("restart")),
      slider, clock, speedSel);

    main.replaceChildren(head,
      h("section", { class: "stage" },
        h("div", { class: "panel plan" },
          h("div", { class: "panel-head" }, h("span", { class: "label", text: "plan view" }),
            h("span", { class: "meta", text: "replayed from the log · oriented like the fixed camera" })),
          plan.svg, planLegend(ep.scene)),
        h("div", { class: "panel" },
          h("div", { class: "panel-head" }, h("span", { class: "label", text: "cameras" }),
            h("span", { class: "meta", text: Object.keys(ep.frames || {}).length ? "the latest frame at the playhead" : "" })),
          cams.node)),
      h("section", { class: "panel trace" }, transport, h("div", { class: "trace-wrap" }, trace.svg),
        h("div", { class: "legend" },
          h("span", {}, h("i", { style: "background:var(--ink-2);opacity:.55" }), "model turn: picture taken → answer landed"),
          h("span", {}, h("i", { style: "background:var(--act-move)" }), "move"),
          h("span", {}, h("i", { style: "background:var(--act-grip)" }), "gripper"),
          h("span", {}, h("i", { style: "background:var(--fail)" }), "failed action (hover for why)"),
          h("span", {}, h("i", { style: "background:var(--warn)" }), "event"),
          h("span", {}, h("i", { style: "background:var(--ok)" }), "milestone / correct"))),
      details.node);

    function setTime(v) {
      t = clamp(v, 0, ep.duration);
      slider.value = String(t);
      clock.replaceChildren(h("b", { text: secs(t) }), ` / ${secs(ep.duration)}`);
      plan.setTime(t); cams.setTime(t); trace.setTime(t); details.setTime(t);
    }
    function tick(now) {
      if (!playing) return;
      const dt = last ? (now - last) / 1000 : 0;
      last = now;
      if (t >= ep.duration) { pause(); return; }
      setTime(t + dt * speed);
      frame = requestAnimationFrame(tick);
    }
    function play() {
      if (t >= ep.duration) setTime(0);
      playing = true; last = 0;
      playBtn.replaceChildren(icon("pause")); playBtn.setAttribute("aria-label", "Pause");
      frame = requestAnimationFrame(tick);
    }
    function pause() {
      playing = false; cancelAnimationFrame(frame);
      playBtn.replaceChildren(icon("play")); playBtn.setAttribute("aria-label", "Play");
    }
    stopPlayback = pause;
    showEpisode.seek = setTime;
    document.onkeydown = (e) => {
      if (e.target.closest && e.target.closest("input[type=search], select")) return;
      if (e.key === " ") { e.preventDefault(); playing ? pause() : play(); }
      else if (e.key === "ArrowRight") { pause(); setTime(t + 0.5); }
      else if (e.key === "ArrowLeft") { pause(); setTime(t - 0.5); }
    };
    setTime(0);
  }

  // ------------------------------------------------------------ rail

  function buildEpisodes(root) {
    const eps = DATA.episodes;
    const main = h("main", { class: "main", id: "episode" });
    if (!eps.length) {
      main.append(h("div", { class: "empty" }, h("b", { text: "No episodes in this file" }),
        "Run some with ", h("code", { text: "ruibench run" }), " and open this page again."));
      root.append(main);
      return;
    }
    const search = h("input", { type: "search", id: "filter", placeholder: "Filter: task, level, seed…", "aria-label": "Filter episodes" });
    const failuresOnly = h("input", { type: "checkbox", id: "failures" });
    const groups = h("div", { class: "groups" });
    const select = h("select", { id: "episode-select", "aria-label": "Episode" });
    const buttons = new Map();
    const vary = (list, key) => new Set(list.map((e) => e[key])).size > 1;

    const byTask = new Map();
    for (const ep of eps) {
      if (!byTask.has(ep.task)) byTask.set(ep.task, []);
      byTask.get(ep.task).push(ep);
    }
    for (const [task, list] of byTask) {
      const g = h("div", { class: "group", "data-task": task },
        h("h3", { class: "label", text: `${task} · ${list[0].name}` }));
      const showIface = vary(list, "interface"), showAgent = vary(list, "agent"), showCond = vary(list, "condition");
      for (const ep of list) {
        const extra = [showIface && ep.interface, showAgent && ep.agent, showCond && ep.condition].filter(Boolean).join(" · ");
        const btn = h("button", { type: "button", class: "ep", "data-id": ep.id, onclick: () => select_(ep.id) },
          h("span", { class: `dot ${ep.outcome.success ? "ok" : "fail"}`, title: ep.outcome.success ? "success" : ep.outcome.failure }),
          h("span", {}, `${ep.level} · seed ${ep.seed}`, extra ? h("div", { class: "meta", text: extra }) : null),
          h("span", { class: "meta", text: `${Math.round(ep.outcome.progress * 100)}%` }));
        btn.dataset.search = `${ep.task} ${ep.name} ${ep.level} seed ${ep.seed} ${ep.interface} ${ep.agent} ${ep.condition} ${ep.outcome.failure}`.toLowerCase();
        buttons.set(ep.id, btn);
        g.append(btn);
        select.append(h("option", { value: ep.id, text: `${ep.task} · ${ep.level} · seed ${ep.seed}${extra ? ` · ${extra}` : ""} · ${ep.outcome.success ? "✓" : "✗"}` }));
      }
      groups.append(g);
    }
    function applyFilter() {
      const q = search.value.trim().toLowerCase();
      for (const g of groups.children) {
        let any = false;
        for (const btn of g.querySelectorAll(".ep")) {
          const ep = eps.find((e) => e.id === btn.dataset.id);
          const show = (!q || btn.dataset.search.includes(q)) && (!failuresOnly.checked || !ep.outcome.success);
          btn.hidden = !show;
          any = any || show;
        }
        g.hidden = !any;
      }
    }
    search.addEventListener("input", applyFilter);
    failuresOnly.addEventListener("change", applyFilter);
    select.addEventListener("change", () => select_(select.value));

    function select_(id) {
      const ep = eps.find((e) => e.id === id) || eps[0];
      for (const [k, btn] of buttons) btn.setAttribute("aria-current", String(k === ep.id));
      select.value = ep.id;
      history.replaceState(null, "", `#ep=${encodeURIComponent(ep.id)}`);
      showEpisode(ep, main);
    }
    const ok = eps.filter((e) => e.outcome.success).length;
    const rail = h("aside", { class: "rail", "aria-label": "Episodes" },
      h("div", { class: "label", text: `${eps.length} episodes · ${ok} succeeded` }),
      search,
      h("label", { class: "toggle", for: "failures" }, failuresOnly, "failures only"),
      select, groups);
    root.append(h("div", { class: "shell" }, rail, main));
    const wanted = decodeURIComponent((INITIAL_HASH.match(/ep=([^&]+)/) || [])[1] || "");
    const at = Number((INITIAL_HASH.match(/[#&]t=([0-9.]+)/) || [])[1]);
    select_(eps.some((e) => e.id === wanted) ? wanted : eps[0].id);
    if (at > 0 && showEpisode.seek) showEpisode.seek(at);          // #ep=…&t=12.5 opens at that moment
  }

  // ------------------------------------------------------------ boot
  const app = document.getElementById("app");
  app.append(h("div", { class: "topbar" },
    h("div", { class: "brand" }, h("b", { text: "RUIBench" }), h("span", { text: DATA.title }))));
  const pane = h("div", {});
  buildEpisodes(pane);
  app.append(pane);
})();
