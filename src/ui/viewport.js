// Side view on graph paper: 1 m major grid, terrain profile, live robot,
// dashed ghost of a recorded run, and push arrows.

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

function fit(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

export class Viewport {
  constructor(canvas) {
    this.canvas = canvas;
    this.profile = null;
    this.planes = [0];
    this.camX = 0;
    this.color = '#0072b2';
    this.message = 'Load a model to begin.';
  }

  setGround(profile, planes) { this.profile = profile; this.planes = planes ?? [0]; }

  draw(frame) {
    const { ctx, w, h } = fit(this.canvas);
    const scale = h / 1.9; // px per metre
    if (frame) this.camX += (frame.torsoX - this.camX) * 0.25;
    const ox = w * 0.4 - this.camX * scale, oz = h * 0.82;
    const X = (x) => ox + x * scale, Z = (z) => oz - z * scale;
    ctx.clearRect(0, 0, w, h);

    // Graph paper.
    const x0 = (0 - ox) / scale, x1 = (w - ox) / scale;
    ctx.lineWidth = 1;
    for (let gx = Math.floor(x0 * 10) / 10; gx <= x1; gx += 0.1) {
      const major = Math.abs(gx - Math.round(gx)) < 1e-6;
      ctx.strokeStyle = major ? css('--grid-major') : css('--grid');
      ctx.beginPath(); ctx.moveTo(Math.round(X(gx)) + 0.5, 0); ctx.lineTo(Math.round(X(gx)) + 0.5, h); ctx.stroke();
      if (major) {
        ctx.fillStyle = css('--ink-soft'); ctx.font = '12px ' + css('--font');
        ctx.fillText(`${Math.round(gx)} m`, X(gx) + 3, h - 6);
      }
    }
    for (let gz = -1; gz <= 2; gz += 0.1) {
      const major = Math.abs(gz - Math.round(gz)) < 1e-6;
      ctx.strokeStyle = major ? css('--grid-major') : css('--grid');
      ctx.beginPath(); ctx.moveTo(0, Math.round(Z(gz)) + 0.5); ctx.lineTo(w, Math.round(Z(gz)) + 0.5); ctx.stroke();
    }

    // Ground.
    ctx.strokeStyle = css('--soil'); ctx.fillStyle = css('--soil-fill'); ctx.lineWidth = 2;
    ctx.beginPath();
    if (this.profile && this.profile.x.length) {
      const { x, z } = this.profile;
      let started = false;
      for (let i = 0; i < x.length; i++) {
        if (x[i] < x0 - 1 || x[i] > x1 + 1) continue;
        if (!started) { ctx.moveTo(X(x[i]), Z(z[i])); started = true; } else ctx.lineTo(X(x[i]), Z(z[i]));
      }
      ctx.stroke(); ctx.lineTo(w, h); ctx.lineTo(0, h); ctx.closePath(); ctx.fill();
    } else {
      const z = this.planes[0] ?? 0;
      ctx.moveTo(0, Z(z)); ctx.lineTo(w, Z(z)); ctx.stroke();
      ctx.fillRect(0, Z(z), w, h - Z(z));
    }

    if (!frame) {
      ctx.fillStyle = css('--ink-soft'); ctx.font = '15px ' + css('--font');
      ctx.fillText(this.message, 16, 32);
      return;
    }

    const drawSegs = (segs, stroke, dashed) => {
      ctx.lineCap = 'round';
      for (let i = 0; i < segs.length; i += 5) {
        const r = Math.max(1.5, segs[i + 4] * scale);
        ctx.beginPath();
        ctx.moveTo(X(segs[i]), Z(segs[i + 1]));
        ctx.lineTo(X(segs[i + 2]), Z(segs[i + 3]));
        if (dashed) {
          ctx.setLineDash([6, 5]); ctx.lineWidth = 2; ctx.strokeStyle = stroke; ctx.stroke(); ctx.setLineDash([]);
        } else {
          ctx.lineWidth = 2 * r; ctx.strokeStyle = stroke; ctx.stroke();
        }
      }
    };
    if (frame.ghost) drawSegs(frame.ghost, css('--ghost'), true);
    drawSegs(frame.segs, this.color, false);

    if (frame.push) {
      const [fx, , fz, , ty] = frame.push;
      const cx = X(frame.torsoX), cz = Z(0.75);
      ctx.strokeStyle = css('--push'); ctx.fillStyle = css('--push'); ctx.lineWidth = 3;
      const mag = Math.hypot(fx, fz);
      if (mag > 0) {
        const len = 30 + Math.min(70, mag / 6), ux = fx / mag, uz = -fz / mag;
        const sx = cx - ux * len, sz = cz - uz * len;
        ctx.beginPath(); ctx.moveTo(sx, sz); ctx.lineTo(cx, cz); ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(cx, cz);
        ctx.lineTo(cx - ux * 10 - uz * 6, cz - uz * 10 + ux * 6);
        ctx.lineTo(cx - ux * 10 + uz * 6, cz - uz * 10 - ux * 6);
        ctx.fill();
      }
      if (ty) {
        // +ty pitches the nose (+x) down: clockwise on screen.
        const r = 18, ax = cx, az = cz - 45, cw = ty > 0;
        const a0 = Math.PI, a1 = cw ? Math.PI + 1.5 * Math.PI : Math.PI - 1.5 * Math.PI;
        ctx.beginPath(); ctx.arc(ax, az, r, a0, a1, !cw); ctx.stroke();
        const ex = ax + r * Math.cos(a1), ez = az + r * Math.sin(a1);
        const tx = cw ? -Math.sin(a1) : Math.sin(a1), tz = cw ? Math.cos(a1) : -Math.cos(a1);
        ctx.beginPath();
        ctx.moveTo(ex + tx * 8, ez + tz * 8);
        ctx.lineTo(ex - tz * 6, ez + tx * 6);
        ctx.lineTo(ex + tz * 6, ez - tx * 6);
        ctx.fill();
      }
    }

    ctx.fillStyle = css('--ink'); ctx.font = '14px ' + css('--font');
    ctx.fillText(`t = ${frame.t.toFixed(2)} s`, w - 110, 22);
  }
}

export class Strip {
  constructor(canvas, seconds = 6) { this.canvas = canvas; this.seconds = seconds; this.pts = []; }
  reset() { this.pts = []; }
  push(t, speed, target) {
    if (this.pts.length && t < this.pts[this.pts.length - 1][0]) this.pts = [];
    this.pts.push([t, speed, target]);
    while (this.pts.length && this.pts[0][0] < t - this.seconds) this.pts.shift();
  }
  draw(color) {
    const { ctx, w, h } = fit(this.canvas);
    ctx.clearRect(0, 0, w, h);
    if (!this.pts.length) return;
    const tEnd = this.pts[this.pts.length - 1][0], tStart = tEnd - this.seconds;
    const vmax = Math.max(4, ...this.pts.map((p) => Math.max(p[1], p[2]))) * 1.1;
    const vmin = Math.min(0, ...this.pts.map((p) => p[1]));
    const X = (t) => ((t - tStart) / this.seconds) * w, Y = (v) => h - 14 - ((v - vmin) / (vmax - vmin)) * (h - 20);
    ctx.strokeStyle = css('--grid-major'); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, Y(0)); ctx.lineTo(w, Y(0)); ctx.stroke();
    const line = (k, stroke, dash) => {
      ctx.setLineDash(dash); ctx.strokeStyle = stroke; ctx.lineWidth = 2; ctx.beginPath();
      this.pts.forEach((p, i) => (i ? ctx.lineTo(X(p[0]), Y(p[k])) : ctx.moveTo(X(p[0]), Y(p[k]))));
      ctx.stroke(); ctx.setLineDash([]);
    };
    line(2, css('--ink-soft'), [5, 4]);
    line(1, color, []);
    ctx.fillStyle = css('--ink-soft'); ctx.font = '12px ' + css('--font');
    ctx.fillText(`${vmax.toFixed(1)} m/s`, 4, 12);
  }
}
