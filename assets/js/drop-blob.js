/*!
 * drop-blob.js — an interactive liquid-gradient droplet, in one <canvas>.
 * Vanilla JS, no dependencies, no build step. MIT licensed — use freely.
 *
 *   <div id="blob" style="max-width:240px; aspect-ratio:1/1"></div>
 *   <script src="/assets/js/drop-blob.js"></script>
 *   <script>DropBlob('#blob');</script>
 *
 * How it works, in four lines:
 *   shape  — smooth union (smin) of two circles => a teardrop SDF
 *   motion — fbm noise warps the sampling coordinate (domain warping)
 *   bleed  — a WIDE smoothstep band on the SDF, not a hard edge
 *   touch  — a gaussian bump at the pointer, added into the SDF
 */
(function (global, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else if (typeof define === 'function' && define.amd) define(factory);
  else global.DropBlob = factory();
})(typeof window !== 'undefined' ? window : this, function () {
  'use strict';

  var DEFAULTS = {
    // Four gradient stops. Keep the hues far apart or they mix into grey mud.
    colors: ['#FF6B2C', '#E0186B', '#4B2ED6', '#16BFB4'],
    scale: 1.1,      // noise frequency — higher = busier, more marbled
    warp: 0.18,      // domain-warp amount — how irregularly the edge breathes
    soft: 0.16,      // BLEED WIDTH. 0.02 = hard plastic, 0.40 = fog. Start here.
    speed: 1.0,      // 0 freezes it on a still frame
    grain: 0.035,    // film grain; kills 8-bit banding on large gradients
    pull: 0.13,      // pointer gravity: >0 attracts, 0 off, <0 repels
    fit: 'contain',  // 'contain' keeps the droplet whole; 'width' fills a wide
                     // banner and crops top/bottom into a liquid band
    zoom: 1.0,       // >1 enlarges the droplet inside its box
    interactive: true,
    maxDPR: 2,       // cap devicePixelRatio — 3x on a phone is wasted fill rate
    autoPause: true, // stop rendering when scrolled out of view or tab hidden
    respectReducedMotion: true,
    fallback: true   // paint a static CSS gradient if WebGL is unavailable
  };

  var VERT = 'attribute vec2 p;void main(){gl_Position=vec4(p,0.0,1.0);}';

  var FRAG = [
    'precision highp float;',
    'uniform vec2 uRes, uMouse;',
    'uniform float uTime, uScale, uWarp, uSoft, uGrain, uPull, uNorm;',
    'uniform vec3 uC1, uC2, uC3, uC4;',

    'float hash(vec2 p){ p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }',

    'float vnoise(vec2 p){',
    '  vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);',
    '  float a = hash(i), b = hash(i + vec2(1.0, 0.0));',
    '  float c = hash(i + vec2(0.0, 1.0)), d = hash(i + vec2(1.0, 1.0));',
    '  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);',
    '}',

    'float fbm(vec2 p){',
    '  float v = 0.0, a = 0.5;',
    '  for (int i = 0; i < 5; i++){ v += a * vnoise(p); p *= 2.03; a *= 0.5; }',
    '  return v;',
    '}',

    // smooth minimum: the whole reason this reads as liquid and not as two circles
    'float smin(float a, float b, float k){',
    '  float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);',
    '  return mix(b, a, h) - k * h * (1.0 - h);',
    '}',

    // big circle + small circle above it, smoothly fused => teardrop
    'float drop(vec2 p){',
    '  return smin(length(p - vec2(0.0, -0.06)) - 0.40,',
    '              length(p - vec2(0.0,  0.46)) - 0.075, 0.26);',
    '}',

    'void main(){',
    '  vec2 uv = (gl_FragCoord.xy * 2.0 - uRes) / uNorm;',
    '  float t = uTime * 0.06;',

    // two rounds of domain warping — this is where the slow drift comes from
    '  vec2 q = vec2(fbm(uv * uScale + vec2(0.0, t)),',
    '                fbm(uv * uScale + vec2(5.2, 1.3 - t)));',
    '  vec2 r = vec2(fbm(uv * uScale + 4.0 * q + vec2(1.7, 9.2) + t),',
    '                fbm(uv * uScale + 4.0 * q + vec2(8.3, 2.8) - t));',
    '  float f = fbm(uv * uScale + 4.0 * r);',

    // pointer: a gaussian bump subtracted from the distance field
    '  float md = length(uv - uMouse);',
    '  float bump = exp(-md * md * 7.0) * uPull;',
    '  float d = drop(uv + (r - 0.5) * uWarp) - bump;',

    // the bleed: a wide transition band instead of a hard cut
    '  float mask = 1.0 - smoothstep(-uSoft, uSoft, d);',

    // colour: the same noise field, used as a lookup index
    '  float m = clamp(f * 2.0 - 0.35 + bump * 1.6, 0.0, 1.0);',
    '  vec3 col = mix(uC3, uC2, smoothstep(0.0, 0.45, m));',
    '  col = mix(col, uC1, smoothstep(0.35, 0.78, m));',
    '  col = mix(col, uC4, smoothstep(0.80, 1.0, m));',
    '  col += (hash(gl_FragCoord.xy + fract(uTime) * 91.7) - 0.5) * uGrain;',

    // non-premultiplied alpha, so it composites over any page background
    '  gl_FragColor = vec4(col, mask);',
    '}'
  ].join('\n');

  function hex2rgb(h) {
    h = String(h).trim().replace('#', '');
    if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
    var n = parseInt(h, 16);
    return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
  }

  function compile(gl, type, src) {
    var s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      console.warn('[drop-blob] shader:', gl.getShaderInfoLog(s));
      gl.deleteShader(s);
      return null;
    }
    return s;
  }

  function DropBlob(target, options) {
    if (!(this instanceof DropBlob)) return new DropBlob(target, options);

    var el = typeof target === 'string' ? document.querySelector(target) : target;
    if (!el) { console.warn('[drop-blob] target not found:', target); return; }

    var o = {};
    for (var k in DEFAULTS) o[k] = DEFAULTS[k];
    for (var k2 in (options || {})) o[k2] = options[k2];

    // A container with no height would render a zero-pixel canvas.
    if (!el.clientHeight) el.style.aspectRatio = el.style.aspectRatio || '1 / 1';
    if (getComputedStyle(el).position === 'static') el.style.position = 'relative';

    var canvas = document.createElement('canvas');
    canvas.setAttribute('aria-hidden', 'true');
    canvas.style.cssText = 'display:block;width:100%;height:100%;';
    el.appendChild(canvas);

    var gl = canvas.getContext('webgl', {
      alpha: true, premultipliedAlpha: false, antialias: true, depth: false
    }) || canvas.getContext('experimental-webgl', {
      alpha: true, premultipliedAlpha: false, antialias: true, depth: false
    });

    if (!gl) {
      canvas.remove();
      if (o.fallback) {
        // Not a real substitute, but far better than an empty hole.
        el.style.background =
          'radial-gradient(60% 55% at 38% 34%, ' + o.colors[0] + ' 0%, transparent 62%),' +
          'radial-gradient(55% 60% at 66% 46%, ' + o.colors[1] + ' 0%, transparent 64%),' +
          'radial-gradient(70% 70% at 46% 68%, ' + o.colors[2] + ' 0%, transparent 66%),' +
          'radial-gradient(60% 60% at 62% 74%, ' + o.colors[3] + ' 0%, transparent 62%)';
        el.style.filter = 'blur(14px)';
        el.style.borderRadius = '50% 50% 50% 50% / 60% 60% 40% 40%';
      }
      this.destroy = function () { el.style.background = el.style.filter = ''; };
      this.pause = this.resume = function () {};
      this.set = function () {};
      return;
    }

    var prog = gl.createProgram();
    var vs = compile(gl, gl.VERTEX_SHADER, VERT);
    var fs = compile(gl, gl.FRAGMENT_SHADER, FRAG);
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    gl.useProgram(prog);

    // one full-screen triangle, not two — fewer vertices, no diagonal seam
    var buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    var loc = gl.getAttribLocation(prog, 'p');
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

    var U = {};
    ['uRes', 'uMouse', 'uTime', 'uScale', 'uWarp', 'uSoft', 'uGrain', 'uPull', 'uNorm',
      'uC1', 'uC2', 'uC3', 'uC4'].forEach(function (n) {
      U[n] = gl.getUniformLocation(prog, n);
    });
    gl.clearColor(0, 0, 0, 0);

    function pushColors() {
      gl.uniform3fv(U.uC1, hex2rgb(o.colors[0]));
      gl.uniform3fv(U.uC2, hex2rgb(o.colors[1]));
      gl.uniform3fv(U.uC3, hex2rgb(o.colors[2]));
      gl.uniform3fv(U.uC4, hex2rgb(o.colors[3]));
    }
    pushColors();

    // ---- state -------------------------------------------------------------
    var clock = 0, last = 0, raf = 0, running = false, destroyed = false;
    var onScreen = true, tabVisible = true, userPaused = false;
    var ptr = { tx: 0, ty: 0, cx: 0, cy: 0, on: 0, amt: 0 };
    var reduced = o.respectReducedMotion && window.matchMedia &&
                  window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    function resize() {
      var dpr = Math.min(window.devicePixelRatio || 1, o.maxDPR);
      var w = Math.max(1, Math.round(el.clientWidth * dpr));
      var h = Math.max(1, Math.round(el.clientHeight * dpr));
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w; canvas.height = h;
        gl.viewport(0, 0, w, h);
      }
    }

    // Which dimension the droplet is measured against.
    function normOf() {
      var w = canvas.width, h = canvas.height;
      return (o.fit === 'width' ? Math.max(w, h) : Math.min(w, h)) * (o.zoom || 1);
    }

    function render(dt) {
      resize();
      clock += dt * 0.001 * o.speed;

      // Pointer coords arrive as jumps. Interpolating every frame IS the
      // viscosity — without this the droplet twitches.
      ptr.cx += (ptr.tx - ptr.cx) * 0.08;
      ptr.cy += (ptr.ty - ptr.cy) * 0.08;
      ptr.amt += (ptr.on - ptr.amt) * 0.05;   // and this is why it doesn't snap back

      gl.uniform2f(U.uRes, canvas.width, canvas.height);
      gl.uniform1f(U.uNorm, normOf());
      gl.uniform2f(U.uMouse, ptr.cx, ptr.cy);
      gl.uniform1f(U.uTime, clock);
      gl.uniform1f(U.uScale, o.scale);
      gl.uniform1f(U.uWarp, o.warp);
      gl.uniform1f(U.uSoft, o.soft);
      gl.uniform1f(U.uGrain, o.grain);
      gl.uniform1f(U.uPull, o.pull * ptr.amt);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }

    function frame(t) {
      if (destroyed) return;
      var dt = last ? Math.min(t - last, 64) : 16;  // clamp: tab-switch time jumps
      last = t;
      render(dt);
      raf = requestAnimationFrame(frame);
    }

    function shouldRun() {
      return !destroyed && !reduced && !userPaused && onScreen && tabVisible;
    }
    function sync() {
      if (shouldRun() && !running) {
        running = true; last = 0; raf = requestAnimationFrame(frame);
      } else if (!shouldRun() && running) {
        running = false; cancelAnimationFrame(raf);
      }
    }

    // ---- listeners ---------------------------------------------------------
    var ro = null, io = null;
    if (window.ResizeObserver) { ro = new ResizeObserver(function () { render(0); }); ro.observe(el); }
    else window.addEventListener('resize', onWinResize, { passive: true });
    function onWinResize() { render(0); }

    if (o.autoPause && window.IntersectionObserver) {
      io = new IntersectionObserver(function (es) {
        onScreen = es[0].isIntersecting; sync();
      }, { rootMargin: '150px' });
      io.observe(el);
    }
    function onVis() { tabVisible = !document.hidden; sync(); }
    if (o.autoPause) document.addEventListener('visibilitychange', onVis);

    function onMove(e) {
      var r = el.getBoundingClientRect();
      if (!r.width || !r.height) return;
      var nx = (e.clientX - r.left) / r.width, ny = (e.clientY - r.top) / r.height;
      var w = canvas.width, h = canvas.height, s = normOf();
      ptr.tx = (nx * w * 2 - w) / s;
      ptr.ty = ((1 - ny) * h * 2 - h) / s;   // GL's Y points up
      ptr.on = 1;
    }
    function onLeave() { ptr.on = 0; }
    if (o.interactive) {
      // pointer* covers mouse, pen and touch; passive so it never eats a scroll
      el.addEventListener('pointermove', onMove, { passive: true });
      el.addEventListener('pointerleave', onLeave);
      el.addEventListener('pointercancel', onLeave);
    }

    function onLost(e) { e.preventDefault(); running = false; cancelAnimationFrame(raf); }
    canvas.addEventListener('webglcontextlost', onLost);

    // ---- go ----------------------------------------------------------------
    render(0);   // always paint one still frame, even under reduced-motion
    sync();

    // ---- public API --------------------------------------------------------
    this.pause = function () { userPaused = true; sync(); };
    this.resume = function () { userPaused = false; sync(); };
    this.set = function (next) {
      for (var n in (next || {})) o[n] = next[n];
      if (next && next.colors) pushColors();
      if (!running) render(0);
      return this;
    };
    this.destroy = function () {
      destroyed = true;
      cancelAnimationFrame(raf);
      if (ro) ro.disconnect(); else window.removeEventListener('resize', onWinResize);
      if (io) io.disconnect();
      document.removeEventListener('visibilitychange', onVis);
      el.removeEventListener('pointermove', onMove);
      el.removeEventListener('pointerleave', onLeave);
      el.removeEventListener('pointercancel', onLeave);
      canvas.removeEventListener('webglcontextlost', onLost);
      var ext = gl.getExtension('WEBGL_lose_context');
      if (ext) ext.loseContext();
      canvas.remove();
    };
    this.el = el;
    this.canvas = canvas;
  }

  DropBlob.defaults = DEFAULTS;
  return DropBlob;
});
