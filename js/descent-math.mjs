export const BOUND = 3;

export const PRESETS = {
  saddle: {
    name: 'Saddle', start: [1.8, 0.12],
    description: 'L = x² − y². The origin is flat but not a minimum. Approach along x, then escape along y. Unbounded below; no global minimum.',
    minima: [],
    sample: (x, y) => ({ loss: x * x - y * y, gradient: [2 * x, -2 * y] }),
  },
  valley: {
    name: 'Rosenbrock valley', start: [-1.2, 1.5],
    description: 'L = 0.02[(1 − x)² + 100(y − x²)²]. A steep-sided, curved valley leads to (1, 1). Watch momentum overshoot the bend.',
    minima: [{ x: 1, y: 1, loss: 0, global: true }],
    sample: (x, y) => ({ loss: 0.02 * ((1 - x) ** 2 + 100 * (y - x * x) ** 2), gradient: [0.04 * (x - 1) - 8 * x * (y - x * x), 4 * (y - x * x)] }),
  },
  wells: {
    name: 'Double well', start: [0.12, 1.8],
    description: 'L = (x² − 1)² + 0.5y². Two equally deep global minima at (−1, 0) and (1, 0), separated by a saddle. Try starting on either side of x = 0.',
    minima: [{ x: -1, y: 0, loss: 0, global: true }, { x: 1, y: 0, loss: 0, global: true }],
    sample: (x, y) => ({ loss: (x * x - 1) ** 2 + 0.5 * y * y, gradient: [4 * x * (x * x - 1), y] }),
  },
  bowl: {
    name: 'Elongated bowl', start: [2.4, 1.8],
    description: 'L = 0.08x² + 2y². Curvature differs by 25× between the axes. Compare SGD’s slow crawl with adaptive updates; try SGD at η = 0.45 for zigzags.',
    minima: [{ x: 0, y: 0, loss: 0, global: true }],
    sample: (x, y) => ({ loss: 0.08 * x * x + 2 * y * y, gradient: [0.16 * x, 4 * y] }),
  },
};

export function randomFromSeed(seed) {
  let value = seed >>> 0;
  return () => {
    value += 0x6d2b79f5;
    let t = Math.imul(value ^ (value >>> 15), 1 | value);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// A random curved valley with cosine-connected wells. Monotone segments
// between alternating minima/maxima guarantee the requested number of minima.
// Rotation and the invertible (u, v - bend(u)) mapping preserve that count.
export function makeSurface(seed, localMinima) {
  const random = randomFromSeed(seed);
  const total = localMinima + 1;
  const angle = random() * Math.PI * 2;
  const c = Math.cos(angle), s = Math.sin(angle);
  const phase = random() * Math.PI * 2;
  const frequency = 0.6 + random() * 0.5;
  const curvature = 0.35 + random() * 0.5;
  const bend = u => 0.55 * Math.sin(frequency * u + phase);
  const bendDerivative = u => 0.55 * frequency * Math.cos(frequency * u + phase);
  const globalIndex = Math.floor(random() * total);
  const wells = Array.from({ length: total }, (_, i) => ({
    u: total === 1 ? (random() - 0.5) : -2.15 + i * 4.3 / (total - 1) + (random() - 0.5) * 0.2,
    z: i === globalIndex ? 0 : 0.25 + random() * 0.9,
    global: i === globalIndex,
  }));
  const knots = [];
  wells.forEach((well, i) => {
    if (i) knots.push({ u: (wells[i - 1].u + well.u) / 2, z: 1.8 + random() * 1.7 });
    knots.push(well);
  });
  function profile(u) {
    const first = knots[0], last = knots.at(-1);
    if (u <= first.u) return [first.z + 0.9 * (u - first.u) ** 2, 1.8 * (u - first.u)];
    if (u >= last.u) return [last.z + 0.9 * (u - last.u) ** 2, 1.8 * (u - last.u)];
    const i = knots.findIndex(knot => knot.u >= u);
    const a = knots[i - 1], b = knots[i];
    const width = b.u - a.u, t = (u - a.u) / width;
    return [a.z + (b.z - a.z) * (1 - Math.cos(Math.PI * t)) / 2,
      (b.z - a.z) * Math.PI * Math.sin(Math.PI * t) / (2 * width)];
  }
  function sample(x, y) {
    const u = c * x + s * y, v = -s * x + c * y;
    const [base, derivative] = profile(u);
    const offset = v - bend(u);
    const du = derivative - 2 * curvature * offset * bendDerivative(u);
    const dv = 2 * curvature * offset;
    return { loss: base + curvature * offset ** 2, gradient: [c * du - s * dv, s * du + c * dv] };
  }
  return { sample, minima: wells.map(well => ({
    x: c * well.u - s * bend(well.u), y: s * well.u + c * bend(well.u),
    loss: well.z, global: well.global,
  })) };
}

export function freshState() { return { step: 0, m: [0, 0], v: [0, 0] }; }

export function optimizerStep(point, gradient, state, options) {
  const { optimizer, lr, momentum = 0.9, beta1 = 0.9, beta2 = 0.999, decay = 0.99, epsilon = 1e-8 } = options;
  const next = { step: state.step + 1, m: [...state.m], v: [...state.v] };
  const position = point.map((value, i) => {
    const g = gradient[i];
    let update = g;
    if (optimizer === 'momentum') {
      next.m[i] = momentum * state.m[i] + g;
      update = next.m[i];
    } else if (optimizer === 'adam') {
      next.m[i] = beta1 * state.m[i] + (1 - beta1) * g;
      next.v[i] = beta2 * state.v[i] + (1 - beta2) * g * g;
      update = (next.m[i] / (1 - beta1 ** next.step)) / (Math.sqrt(next.v[i] / (1 - beta2 ** next.step)) + epsilon);
    } else if (optimizer === 'rmsprop') {
      next.v[i] = decay * state.v[i] + (1 - decay) * g * g;
      update = g / (Math.sqrt(next.v[i]) + epsilon);
    } else if (optimizer !== 'sgd') throw new Error('Unknown optimizer');
    return value - lr * update;
  });
  return { position, state: next };
}

export function inBounds(point) {
  return point.every(value => Number.isFinite(value) && Math.abs(value) <= BOUND);
}
