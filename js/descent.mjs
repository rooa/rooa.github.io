import { BOUND, PRESETS, makeSurface, freshState, optimizerStep, inBounds } from './descent-math.mjs?v=orbit-1';

const $ = id => document.getElementById(id);
const settings = $('settings');
const surfaceCanvas = $('surface'), mapCanvas = $('map'), historyCanvas = $('history');
const blue = '#365cba', ink = '#171717', muted = '#62615c', line = '#c9c8c2';
const definitions = {
  sgd: { description: 'Exact-gradient SGD. Each step follows the negative gradient.', fields: [] },
  momentum: { description: 'Accumulate velocity to carry movement through shallow regions.', fields: [['momentum', 'Momentum μ', 0.9, 0, 0.9999]] },
  adam: { description: 'Bias-corrected momentum with a separate adaptive scale for each coordinate.', fields: [['beta1', 'First moment β₁', 0.9, 0, 0.9999], ['beta2', 'Second moment β₂', 0.999, 0, 0.99999], ['epsilon', 'Epsilon ε', 1e-8, 1e-12, 1]] },
  rmsprop: { description: 'Scale each coordinate by its recent root-mean-square gradient.', fields: [['decay', 'Decay α', 0.99, 0, 0.99999], ['epsilon', 'Epsilon ε', 1e-8, 1e-12, 1]] },
};
const parameterValues = {};
let landscape, start = [-1.8, 1.8], point = [...start], state = freshState();
let path = [], timer = null, terminal = false, stableSteps = 0, attempted = null;
let mesh = [], projectedFaces = [], maxHeight = 1, displayFloor = 0;
const displayHeight = loss => Math.log1p(Math.max(0, loss - displayFloor));
let surfaceBase, mapBase;
const size = new Map();
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
let yaw = -35, pitch = 26, autoRotate = !reducedMotion.matches;
let gesture = null, orbitFrame = null, lastOrbitTime = null, surfaceVisible = true;

function redrawCamera() {
  cacheSurface();
  if (surfaceBase && path.length) drawTrajectory(surfaceCanvas, surfaceBase, project);
}

function orbitTick(now) {
  orbitFrame = null;
  if (!autoRotate || document.hidden || !surfaceVisible) { lastOrbitTime = null; return; }
  if (lastOrbitTime === null) lastOrbitTime = now;
  const elapsed = now - lastOrbitTime;
  // 24 frames/s is enough for a slow orbit; optimization has its own clock.
  if (elapsed >= 1000 / 24) {
    yaw = (yaw + Math.min(elapsed, 100) * 0.006) % 360;
    lastOrbitTime = now;
    redrawCamera();
  }
  orbitFrame = requestAnimationFrame(orbitTick);
}

function scheduleOrbit() {
  if (autoRotate && !document.hidden && surfaceVisible && orbitFrame === null) orbitFrame = requestAnimationFrame(orbitTick);
}

function setAutoRotate(enabled) {
  autoRotate = enabled;
  $('auto-rotate').setAttribute('aria-pressed', String(enabled));
  $('auto-rotate').textContent = enabled ? 'Pause rotation' : 'Resume rotation';
  if (orbitFrame !== null) cancelAnimationFrame(orbitFrame);
  orbitFrame = null; lastOrbitTime = null;
  scheduleOrbit();
}

function resetView() {
  setAutoRotate(false);
  yaw = -35; pitch = 26;
  redrawCamera();
}

function optimizerFields() {
  const kind = $('optimizer').value;
  const definition = definitions[kind];
  $('optimizer-params').replaceChildren();
  for (const [key, title, initial, min, max] of definition.fields) {
    const label = document.createElement('label');
    label.htmlFor = key;
    label.textContent = title;
    const input = document.createElement('input');
    Object.assign(input, { id: key, type: 'number', min, max, step: 'any', required: true, value: parameterValues[`${kind}:${key}`] ?? initial });
    label.append(input);
    $('optimizer-params').append(label);
  }
  $('optimizer-description').textContent = definition.description;
}

function options() {
  const optimizer = $('optimizer').value;
  const result = { optimizer, lr: Number($('lr').value) };
  for (const [key] of definitions[optimizer].fields) result[key] = Number($(key).value);
  return result;
}

function setStatus(title, detail, running = false) {
  $('status').textContent = title;
  $('status').dataset.state = running ? 'running' : 'idle';
  $('status-detail').textContent = detail;
}

function stop() {
  clearInterval(timer);
  timer = null;
  settings.querySelectorAll('input, select, button').forEach(input => input.disabled = false);
  syncLandscapeControls();
  $('step').disabled = terminal;
  $('start').textContent = terminal ? '↺ Restart' : state.step ? '▶ Resume' : '▶ Start';
  $('start').classList.remove('running');
}

function reset() {
  terminal = false;
  stop();
  point = [...start];
  state = freshState();
  stableSteps = 0;
  attempted = null;
  path = [{ point: [...point], loss: landscape.sample(...point).loss }];
  $('start').textContent = '▶ Start';
  setStatus('Ready', 'Choose a starting point, then start.');
  render();
}

function syncLandscapeControls() {
  const preset = PRESETS[$('landscape').value];
  $('random-settings').hidden = Boolean(preset);
  $('preset-settings').hidden = !preset;
  $('random-settings').querySelectorAll('input, select, button').forEach(input => input.disabled = Boolean(preset) || Boolean(timer));
  if (preset) $('preset-description').textContent = preset.description;
}

function rebuild() {
  const preset = PRESETS[$('landscape').value];
  landscape = preset || makeSurface(Number($('seed').value), Number($('minima').value));
  $('surface-label').textContent = preset ? preset.name : `Loss landscape / seed ${$('seed').value}`;
  syncLandscapeControls();
  mesh = [];
  maxHeight = 0;
  displayFloor = 0;
  const resolution = 60;
  for (let j = 0; j <= resolution; j++) {
    const row = [];
    for (let i = 0; i <= resolution; i++) {
      const x = -BOUND + 2 * BOUND * i / resolution;
      const y = -BOUND + 2 * BOUND * j / resolution;
      const loss = landscape.sample(x, y).loss;
      row.push({ x, y, loss });
      displayFloor = Math.min(displayFloor, loss);
    }
    mesh.push(row);
  }
  for (const row of mesh) for (const vertex of row) maxHeight = Math.max(maxHeight, displayHeight(vertex.loss));
  $('height-caption').textContent = `x, y ∈ [−3, 3] · height = log(1 + loss${displayFloor < 0 ? ` + ${(-displayFloor).toFixed(2)}` : ''}) · readouts show actual loss`;
  cacheViews();
  reset();
}

function chooseStart(x, y) {
  if (timer) return;
  start = [x, y].map(value => Math.max(-BOUND, Math.min(BOUND, value)));
  // Keep inputs and actual starting coordinates exactly aligned.
  start = start.map(value => Number(value.toFixed(4)));
  $('start-x').value = start[0];
  $('start-y').value = start[1];
  reset();
}

function finish(title, detail) {
  terminal = true;
  stop();
  setStatus(title, detail);
}

function advance() {
  if (terminal || !settings.checkValidity()) return;
  const current = landscape.sample(...point);
  const next = optimizerStep(point, current.gradient, state, options());
  if (!inBounds(next.position)) {
    attempted = next.position;
    finish('Out of bounds', `Step ${next.state.step} left [−3, 3]². Last valid position retained; lower the learning rate and retry.`);
    render();
    return;
  }
  const movement = Math.hypot(...next.position.map((value, i) => value - point[i]));
  point = next.position;
  state = next.state;
  const sample = landscape.sample(...point);
  path.push({ point: [...point], loss: sample.loss });
  stableSteps = Math.hypot(...sample.gradient) < 0.001 && movement < 0.001 ? stableSteps + 1 : 0;
  if (stableSteps >= 5) {
    const nearest = landscape.minima.reduce((best, m) => !best || Math.hypot(m.x - point[0], m.y - point[1]) < Math.hypot(best.x - point[0], best.y - point[1]) ? m : best, null);
    const nearMinimum = nearest && Math.hypot(nearest.x - point[0], nearest.y - point[1]) < 0.03;
    finish(nearMinimum ? (nearest.global ? 'Global minimum reached' : 'Local minimum reached') : 'Stationary point reached', 'Gradient and movement stayed below 0.001 for five steps.');
  } else if (state.step >= 2000) {
    finish('Step limit reached', 'Stopped after 2,000 steps. Adjust the settings and restart.');
  } else if (!timer) setStatus('Paused', 'One step taken. Start to continue at 100 ms per step.');
  render();
}

function fit(canvas) {
  const bounds = canvas.getBoundingClientRect();
  const w = Math.max(1, bounds.width), h = Math.max(1, bounds.height);
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  canvas.getContext('2d').setTransform(dpr, 0, 0, dpr, 0, 0);
  size.set(canvas, { w, h, dpr });
}

function backgroundCanvas(canvas) {
  const base = document.createElement('canvas');
  base.width = canvas.width; base.height = canvas.height;
  const { dpr } = size.get(canvas);
  base.getContext('2d').setTransform(dpr, 0, 0, dpr, 0, 0);
  return base;
}

function project(x, y, loss = displayFloor) {
  const { w, h } = size.get(surfaceCanvas);
  const angle = yaw * Math.PI / 180;
  const elevation = pitch * Math.PI / 180;
  const c = Math.cos(angle), s = Math.sin(angle);
  const horizontal = x * c - y * s, depth = x * s + y * c;
  const height = displayHeight(loss) / maxHeight * 3.2;
  const ce = Math.cos(elevation), se = Math.sin(elevation);
  const scale = Math.min(w / 10, (h - 60) / (8.49 * se + 3.2 * ce));
  return { x: w / 2 + horizontal * scale, y: h / 2 + (1.6 * ce + depth * se - height * ce) * scale, depth: depth * ce + height * se };
}

function strokePath(ctx, points, color, width = 1) {
  if (!points.length) return;
  ctx.beginPath(); points.forEach((p, i) => i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y));
  ctx.strokeStyle = color; ctx.lineWidth = width; ctx.stroke();
}

function symbol(ctx, p, global, radius = 4) {
  ctx.beginPath();
  if (global) { ctx.moveTo(p.x, p.y - radius - 1); ctx.lineTo(p.x + radius + 1, p.y); ctx.lineTo(p.x, p.y + radius + 1); ctx.lineTo(p.x - radius - 1, p.y); ctx.closePath(); }
  else ctx.arc(p.x, p.y, radius, 0, 2 * Math.PI);
  ctx.fillStyle = '#f6f5f1'; ctx.fill(); ctx.strokeStyle = ink; ctx.lineWidth = 1.3; ctx.stroke();
}

function cacheSurface() {
  if (!mesh.length) return;
  surfaceBase = backgroundCanvas(surfaceCanvas);
  const ctx = surfaceBase.getContext('2d');
  const { w, h } = size.get(surfaceCanvas);
  ctx.fillStyle = '#f6f5f1'; ctx.fillRect(0, 0, w, h);
  for (let i = -3; i <= 3; i++) {
    strokePath(ctx, [project(i, -3), project(i, 3)], '#deddd7');
    strokePath(ctx, [project(-3, i), project(3, i)], '#deddd7');
  }
  projectedFaces = [];
  for (let j = 0; j < mesh.length - 1; j++) for (let i = 0; i < mesh.length - 1; i++) {
    const vertices = [mesh[j][i], mesh[j][i + 1], mesh[j + 1][i + 1], mesh[j + 1][i]];
    const projected = vertices.map(v => project(v.x, v.y, v.loss));
    projectedFaces.push({ vertices, projected, depth: projected.reduce((sum, v) => sum + v.depth, 0) / 4, i, j });
  }
  projectedFaces.sort((a, b) => a.depth - b.depth);
  for (const face of projectedFaces) {
    const { projected: points, vertices } = face;
    const height = displayHeight(vertices.reduce((sum, v) => sum + v.loss, 0) / 4) / maxHeight;
    const shade = Math.round(236 - height * 38);
    ctx.beginPath(); points.forEach((p, i) => i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)); ctx.closePath();
    ctx.fillStyle = `rgb(${shade}, ${shade + 1}, ${shade - 3})`; ctx.fill();
    ctx.strokeStyle = `rgb(${shade}, ${shade + 1}, ${shade - 3})`; ctx.lineWidth = .6; ctx.stroke();
    if (face.i % 3 === 0) strokePath(ctx, [points[0], points[3]], '#999b91', .65);
    if (face.j % 3 === 0) strokePath(ctx, [points[0], points[1]], '#999b91', .65);
  }
  ctx.font = '11px Helvetica, Arial, sans-serif'; ctx.fillStyle = muted;
  for (const [x, y, label] of [[3.35, -3, 'x'], [-3, 3.35, 'y'], [-3.25, -3.1, '−3'], [3.2, 3.1, '+3']]) {
    const p = project(x, y); ctx.fillText(label, p.x, p.y + 12);
  }
}

function mapPoint(x, y) {
  const { w, h } = size.get(mapCanvas);
  const side = Math.min(w - 24, h - 24);
  return { x: (w - side) / 2 + (x + 3) / 6 * side, y: 10 + (3 - y) / 6 * side };
}

function cacheMap() {
  if (!mesh.length) return;
  mapBase = backgroundCanvas(mapCanvas);
  const ctx = mapBase.getContext('2d');
  const { w, h } = size.get(mapCanvas);
  ctx.fillStyle = '#f6f5f1'; ctx.fillRect(0, 0, w, h);
  for (let j = 0; j < mesh.length - 1; j++) for (let i = 0; i < mesh.length - 1; i++) {
    const a = mesh[j][i], b = mesh[j][i + 1], c = mesh[j + 1][i + 1], d = mesh[j + 1][i];
    const p = mapPoint(a.x, a.y), q = mapPoint(c.x, c.y);
    const height = displayHeight((a.loss + b.loss + c.loss + d.loss) / 4) / maxHeight;
    const shade = Math.round(245 - height * 48);
    ctx.fillStyle = `rgb(${shade},${shade},${shade - 5})`;
    ctx.fillRect(p.x, q.y, q.x - p.x + .5, p.y - q.y + .5);
    for (let level = 1; level <= 12; level++) {
      const threshold = displayFloor + Math.expm1(level / 13 * maxHeight);
      for (const triangle of [[a, b, c], [a, c, d]]) {
        const intersections = [];
        triangle.forEach((v, index) => {
          const next = triangle[(index + 1) % 3];
          if ((v.loss < threshold) !== (next.loss < threshold)) {
            const t = (threshold - v.loss) / (next.loss - v.loss);
            intersections.push(mapPoint(v.x + (next.x - v.x) * t, v.y + (next.y - v.y) * t));
          }
        });
        if (intersections.length === 2) strokePath(ctx, intersections, '#a8aaa0', .6);
      }
    }
  }
  const a = mapPoint(-3, 3), b = mapPoint(3, -3);
  ctx.strokeStyle = line; ctx.lineWidth = 1; ctx.strokeRect(a.x, a.y, b.x - a.x, b.y - a.y);
  ctx.font = '10px Helvetica, Arial, sans-serif'; ctx.fillStyle = muted;
  ctx.fillText('−3', a.x, b.y + 12); ctx.fillText('x  +3', b.x - 24, b.y + 12);
}

function cacheViews() { cacheSurface(); cacheMap(); }

function drawTrajectory(canvas, base, projection) {
  const ctx = canvas.getContext('2d'), { w, h } = size.get(canvas);
  ctx.clearRect(0, 0, w, h); ctx.drawImage(base, 0, 0, w, h);
  landscape.minima.forEach(m => symbol(ctx, projection(m.x, m.y, m.loss), m.global));
  const points = path.map(entry => projection(...entry.point, entry.loss));
  strokePath(ctx, points, blue, 2);
  if (attempted?.every(Number.isFinite)) {
    const difference = attempted.map((value, i) => value - point[i]);
    let ratio = 1;
    difference.forEach((value, i) => {
      if (value > 0) ratio = Math.min(ratio, (3 - point[i]) / value);
      if (value < 0) ratio = Math.min(ratio, (-3 - point[i]) / value);
    });
    const edge = point.map((value, i) => value + ratio * difference[i]);
    const p = projection(...edge, landscape.sample(...edge).loss);
    ctx.setLineDash([4, 4]); strokePath(ctx, [points.at(-1), p], blue, 1.5); ctx.setLineDash([]);
    strokePath(ctx, [{ x: p.x - 4, y: p.y - 4 }, { x: p.x + 4, y: p.y + 4 }], ink, 2);
    strokePath(ctx, [{ x: p.x + 4, y: p.y - 4 }, { x: p.x - 4, y: p.y + 4 }], ink, 2);
  }
  const first = points[0], last = points.at(-1);
  if (!last) return;
  ctx.beginPath(); ctx.arc(first.x, first.y, 4, 0, Math.PI * 2); ctx.fillStyle = '#f6f5f1'; ctx.fill(); ctx.strokeStyle = blue; ctx.lineWidth = 2; ctx.stroke();
  ctx.beginPath(); ctx.arc(last.x, last.y, 5, 0, Math.PI * 2); ctx.fillStyle = timer ? '#ff5a1f' : blue; ctx.fill(); ctx.strokeStyle = '#f6f5f1'; ctx.lineWidth = 2; ctx.stroke();
}

function drawHistory() {
  const ctx = historyCanvas.getContext('2d'), { w, h } = size.get(historyCanvas);
  ctx.clearRect(0, 0, w, h);
  const max = Math.max(0.001, ...path.map(entry => entry.loss));
  const min = Math.min(0, ...path.map(entry => entry.loss));
  ctx.font = '10px Helvetica, Arial, sans-serif'; ctx.fillStyle = muted;
  ctx.fillText(max.toFixed(2), 0, 12); ctx.fillText(min.toFixed(2), 0, h - 19);
  strokePath(ctx, [{ x: 35, y: 8 }, { x: 35, y: h - 24 }, { x: w - 5, y: h - 24 }], line);
  const points = path.map((entry, i) => ({ x: 35 + i / Math.max(1, path.length - 1) * (w - 40), y: h - 24 - (entry.loss - min) / (max - min) * (h - 32) }));
  strokePath(ctx, points, blue, 1.5);
  ctx.fillText('0', 32, h - 7); ctx.fillText(String(state.step), w - 30, h - 7);
}

function render() {
  if (!surfaceBase || !path.length) return;
  drawTrajectory(surfaceCanvas, surfaceBase, project);
  drawTrajectory(mapCanvas, mapBase, mapPoint);
  drawHistory();
  const sample = landscape.sample(...point);
  $('step-count').textContent = String(state.step).padStart(4, '0');
  $('loss').textContent = sample.loss.toFixed(4);
  $('gradient').textContent = Math.hypot(...sample.gradient).toFixed(4);
  $('position').textContent = point.map(value => value.toFixed(3)).join(', ');
}

// Hit-test the same triangles used to draw the surface, frontmost first.
function barycentric(p, a, b, c) {
  const denominator = (b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y);
  if (Math.abs(denominator) < 1e-8) return null;
  const u = ((b.y - c.y) * (p.x - c.x) + (c.x - b.x) * (p.y - c.y)) / denominator;
  const v = ((c.y - a.y) * (p.x - c.x) + (a.x - c.x) * (p.y - c.y)) / denominator;
  return u >= 0 && v >= 0 && u + v <= 1 ? [u, v, 1 - u - v] : null;
}

function pickSurface(event) {
  if (timer) return;
  const rect = surfaceCanvas.getBoundingClientRect();
  const pointer = { x: event.clientX - rect.left, y: event.clientY - rect.top };
  for (let i = projectedFaces.length - 1; i >= 0; i--) {
    const face = projectedFaces[i];
    for (const indices of [[0, 1, 2], [0, 2, 3]]) {
      const weights = barycentric(pointer, ...indices.map(index => face.projected[index]));
      if (weights) {
        const vertices = indices.map(index => face.vertices[index]);
        chooseStart(weights.reduce((sum, weight, j) => sum + weight * vertices[j].x, 0), weights.reduce((sum, weight, j) => sum + weight * vertices[j].y, 0));
        return;
      }
    }
  }
}

surfaceCanvas.addEventListener('pointerdown', event => {
  if (!event.isPrimary || event.button !== 0 || gesture) return;
  setAutoRotate(false);
  gesture = { id: event.pointerId, x: event.clientX, y: event.clientY, yaw, pitch, dragged: false };
  surfaceCanvas.setPointerCapture(event.pointerId);
});
surfaceCanvas.addEventListener('pointermove', event => {
  if (!gesture || gesture.id !== event.pointerId) return;
  const dx = event.clientX - gesture.x, dy = event.clientY - gesture.y;
  if (Math.hypot(dx, dy) > 5) gesture.dragged = true;
  if (!gesture.dragged) return;
  surfaceCanvas.classList.add('dragging');
  yaw = (gesture.yaw + dx * 0.4) % 360;
  pitch = Math.max(12, Math.min(75, gesture.pitch + dy * 0.25));
  redrawCamera();
});
function endGesture(event) {
  if (!gesture || gesture.id !== event.pointerId) return;
  const pick = event.type === 'pointerup' && !gesture.dragged;
  gesture = null;
  surfaceCanvas.classList.remove('dragging');
  if (surfaceCanvas.hasPointerCapture(event.pointerId)) surfaceCanvas.releasePointerCapture(event.pointerId);
  if (pick) pickSurface(event);
}
surfaceCanvas.addEventListener('pointerup', endGesture);
surfaceCanvas.addEventListener('pointercancel', endGesture);
surfaceCanvas.addEventListener('lostpointercapture', endGesture);
surfaceCanvas.addEventListener('keydown', event => {
  if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home'].includes(event.key)) return;
  event.preventDefault();
  setAutoRotate(false);
  if (event.key === 'Home') { resetView(); return; }
  if (event.key === 'ArrowLeft') yaw -= 5;
  if (event.key === 'ArrowRight') yaw += 5;
  if (event.key === 'ArrowUp') pitch = Math.min(75, pitch + 5);
  if (event.key === 'ArrowDown') pitch = Math.max(12, pitch - 5);
  redrawCamera();
});
mapCanvas.addEventListener('click', event => {
  const rect = mapCanvas.getBoundingClientRect(), a = mapPoint(-3, 3), b = mapPoint(3, -3);
  const px = event.clientX - rect.left, py = event.clientY - rect.top;
  if (px >= a.x && px <= b.x && py >= a.y && py <= b.y) chooseStart(-3 + 6 * (px - a.x) / (b.x - a.x), 3 - 6 * (py - a.y) / (b.y - a.y));
});
settings.addEventListener('submit', event => event.preventDefault());
settings.addEventListener('change', event => {
  if (event.target.id === 'optimizer') optimizerFields();
  if (event.target.id === 'landscape') {
    syncLandscapeControls();
    const preset = PRESETS[$('landscape').value];
    if (preset) {
      start = [...preset.start];
      $('start-x').value = start[0]; $('start-y').value = start[1];
    }
    rebuild();
    return;
  }
  if (!settings.reportValidity()) return;
  const id = event.target.id;
  if (id === 'seed' || id === 'minima') rebuild();
  else if (id === 'start-x' || id === 'start-y') chooseStart(Number($('start-x').value), Number($('start-y').value));
  else {
    parameterValues[`${$('optimizer').value}:${id}`] = Number(event.target.value);
    reset();
  }
});
$('generate').addEventListener('click', () => {
  if (!settings.reportValidity()) return;
  $('seed').value = crypto.getRandomValues(new Uint32Array(1))[0];
  rebuild();
});
$('suggested-start').addEventListener('click', () => {
  const preset = PRESETS[$('landscape').value];
  if (preset) chooseStart(...preset.start);
});
$('start').addEventListener('click', () => {
  if (timer) { stop(); setStatus('Paused', 'Optimizer memory retained. Resume or reset the path.'); render(); return; }
  if (!settings.reportValidity()) return;
  if (terminal) reset();
  timer = setInterval(advance, 100);
  settings.querySelectorAll('input, select, button').forEach(input => input.disabled = true);
  $('step').disabled = true;
  $('start').textContent = '■ Stop';
  $('start').classList.add('running');
  setStatus('Running', 'One optimizer update every 100 ms.', true);
  render();
});
$('step').addEventListener('click', () => { if (settings.reportValidity()) advance(); });
$('reset').addEventListener('click', reset);
$('auto-rotate').addEventListener('click', () => setAutoRotate(!autoRotate));
$('reset-view').addEventListener('click', resetView);
reducedMotion.addEventListener('change', event => { if (event.matches) setAutoRotate(false); });
document.addEventListener('visibilitychange', () => {
  if (document.hidden && timer) { stop(); setStatus('Paused', 'Paused while the page is hidden. Resume when ready.'); render(); }
  lastOrbitTime = null;
  scheduleOrbit();
});
optimizerFields();
const mobileLayout = matchMedia('(max-width: 650px)');
const settingsDrawer = document.querySelector('.settings-drawer');
settingsDrawer.open = !mobileLayout.matches;
mobileLayout.addEventListener('change', event => { settingsDrawer.open = !event.matches; });
[surfaceCanvas, mapCanvas, historyCanvas].forEach(fit);
rebuild();
const observer = new ResizeObserver(() => {
  [surfaceCanvas, mapCanvas, historyCanvas].forEach(fit);
  cacheViews(); render();
});
[surfaceCanvas, mapCanvas, historyCanvas].forEach(canvas => observer.observe(canvas));
const visibilityObserver = new IntersectionObserver(entries => {
  surfaceVisible = entries[0].isIntersecting;
  lastOrbitTime = null;
  scheduleOrbit();
});
visibilityObserver.observe(surfaceCanvas);
setAutoRotate(autoRotate);
