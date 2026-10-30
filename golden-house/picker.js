// Golden House dish picker: choose what is in the draw, choose a randomiser, eat.
(() => {
  const root = document.getElementById("gh");
  if (!root) return;
  const menuUrl = new URL("menu.json", document.currentScript.src);
  const $ = (s) => root.querySelector(s);
  const $$ = (s) => root.querySelectorAll(s);
  const stage = $(".gh-stage");
  const readout = $(".gh-readout");
  const goBtn = $(".gh-go");
  const soundBtn = $(".gh-sound");
  const reduce = matchMedia("(prefers-reduced-motion: reduce)");
  const TAU = Math.PI * 2;
  const mod = (x, m) => ((x % m) + m) % m;
  const rand = (a, b) => a + Math.random() * (b - a);
  const pick = (list) => list[Math.floor(Math.random() * list.length)];
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));
  const css = (name) => getComputedStyle(root).getPropertyValue(name).trim();
  const font = () => getComputedStyle(document.body).fontFamily;
  const el = (tag, cls, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  };
  const store = {
    get(k, d) { try { return localStorage.getItem(k) ?? d; } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch (e) {} },
  };

  // Calls fn(elapsed ms, now) once per frame until it returns false.
  const frames = (fn) => new Promise((done) => {
    const t0 = performance.now();
    const step = (now) => (fn(Math.max(0, now - t0), now) === false ? done() : requestAnimationFrame(step));
    requestAnimationFrame(step);
  });

  function zhNum(n) {
    const d = "零一二三四五六七八九";
    if (n < 10) return d[n];
    if (n < 20) return "十" + (n % 10 ? d[n % 10] : "");
    if (n < 100) return d[Math.floor(n / 10)] + "十" + (n % 10 ? d[n % 10] : "");
    const rest = n % 100;
    const tail = rest === 0 ? "" : rest < 10 ? "零" + d[rest] : rest < 20 ? "一" + zhNum(rest) : zhNum(rest);
    return d[Math.floor(n / 100)] + "百" + tail;
  }

  // --- sound, synthesised in the browser --------------------------------

  const sfx = (() => {
    let ac = null;
    let noise = null;
    let last = 0;
    let on = store.get("gh-sound", "on") !== "off";

    function tone(freq, at, dur, vol, type = "triangle", to = 0) {
      const o = ac.createOscillator();
      const g = ac.createGain();
      o.type = type;
      o.frequency.setValueAtTime(freq, at);
      if (to) o.frequency.exponentialRampToValueAtTime(to, at + dur);
      g.gain.setValueAtTime(vol, at);
      g.gain.exponentialRampToValueAtTime(0.0001, at + dur);
      o.connect(g).connect(ac.destination);
      o.start(at);
      o.stop(at + dur + 0.02);
    }

    function hiss(at, dur, vol, freq, q) {
      if (!noise) {
        noise = ac.createBuffer(1, ac.sampleRate / 2, ac.sampleRate);
        const data = noise.getChannelData(0);
        for (let i = 0; i < data.length; i++) data[i] = Math.random() * 2 - 1;
      }
      const s = ac.createBufferSource();
      const f = ac.createBiquadFilter();
      const g = ac.createGain();
      s.buffer = noise;
      f.type = "bandpass";
      f.frequency.value = freq;
      f.Q.value = q;
      g.gain.setValueAtTime(vol, at);
      g.gain.exponentialRampToValueAtTime(0.0001, at + dur);
      s.connect(f).connect(g).connect(ac.destination);
      s.start(at);
      s.stop(at + dur + 0.02);
    }

    const ready = () => on && ac;
    const now = () => ac.currentTime;

    return {
      get on() { return on; },
      wake() {
        if (!on) return;
        try {
          ac = ac || new (window.AudioContext || window.webkitAudioContext)();
          if (ac.state === "suspended") ac.resume();
        } catch (e) {
          ac = null;
        }
      },
      toggle() {
        on = !on;
        store.set("gh-sound", on ? "on" : "off");
        if (on) this.wake();
        return on;
      },
      tick() {
        if (!ready() || now() - last < 0.03) return;
        last = now();
        tone(1900 + Math.random() * 300, last, 0.035, 0.05);
      },
      clack() { if (ready()) hiss(now(), 0.05, 0.6, rand(1600, 2600), 5); },
      clunk() {
        if (!ready()) return;
        hiss(now(), 0.12, 0.5, 500, 1);
        tone(110, now(), 0.15, 0.2, "sine", 50);
      },
      pop() { if (ready()) tone(380, now(), 0.14, 0.12, "sine", 950); },
      fanfare() {
        if (!ready()) return;
        // Major arpeggio, C E G C.
        [523.25, 659.25, 783.99, 1046.5].forEach((f, i) => tone(f, now() + i * 0.09, i === 3 ? 0.5 : 0.25, 0.07));
      },
    };
  })();

  // --- confetti, rice grains and emoji --------------------------------------

  function burst(x, y, { count = 150, glyphs = [] } = {}) {
    if (reduce.matches) return;
    const cv = el("canvas", "gh-confetti");
    document.body.append(cv);
    const W = innerWidth;
    const H = innerHeight;
    const k = devicePixelRatio || 1;
    cv.width = W * k;
    cv.height = H * k;
    const c = cv.getContext("2d");
    c.scale(k, k);
    c.textAlign = "center";
    c.textBaseline = "middle";
    const colours = [css("--gh-red"), css("--gh-gold"), css("--gh-jade"), css("--gh-light")];
    const life = 2.4;
    const bits = [];
    for (let i = 0; i < count + glyphs.length; i++) {
      const ang = Math.random() * TAU;
      const v = rand(350, 1000);
      const glyph = i >= count ? glyphs[i - count] : null;
      const grain = !glyph && i % 4 === 0;
      bits.push({
        x, y, glyph, grain,
        vx: Math.cos(ang) * v,
        vy: Math.sin(ang) * v - 300,
        rot: Math.random() * TAU,
        vr: rand(-7, 7),
        w: grain ? 6 : rand(6, 11),
        h: grain ? 3 : rand(3, 6),
        size: rand(22, 34),
        colour: grain ? "#fbf7ee" : pick(colours),
      });
    }
    let last = performance.now();
    frames((t, now) => {
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      const age = t / 1000;
      c.clearRect(0, 0, W, H);
      c.globalAlpha = Math.max(0, 1 - Math.max(0, age - (life - 0.8)) / 0.8);
      const drag = Math.pow(0.25, dt);
      for (const b of bits) {
        b.vx *= drag;
        b.vy = b.vy * drag + 1100 * dt;
        b.x += b.vx * dt;
        b.y += b.vy * dt;
        b.rot += b.vr * dt;
        c.save();
        c.translate(b.x, b.y);
        if (b.glyph) {
          c.rotate(b.rot * 0.3);
          c.font = b.size + "px serif";
          c.fillText(b.glyph, 0, 0);
        } else {
          c.rotate(b.rot);
          c.fillStyle = b.colour;
          c.beginPath();
          if (b.grain) c.ellipse(0, 0, b.w / 2, b.h / 2, 0, 0, TAU);
          else c.rect(-b.w / 2, -b.h / 2, b.w, b.h * Math.abs(Math.cos(b.rot * 1.7)) + 0.5);
          c.fill();
        }
        c.restore();
      }
      if (age < life) return true;
      cv.remove();
      return false;
    });
  }

  // --- randomiser 1: the wheel -----------------------------------------------

  function makeWheel() {
    let cv, ctx, pointer;
    let list = [];
    let face = null;
    let size = 0;
    let dpr = 1;
    let theta = 0;      // rotation, clockwise, radians
    let omega = 0;      // rotation in the last frame, for motion blur
    let kick = 0;       // pointer deflection, decays to 0
    let winner = -1;
    let winAt = 0;
    let spinning = false;
    let anim = 0;

    const indexAt = (th) => Math.floor(mod(-th, TAU) / (TAU / list.length)) % list.length;

    function mount(box) {
      const wrap = el("div", "gh-wheelbox");
      pointer = el("div", "gh-pointer");
      cv = el("canvas", "gh-wheel");
      const hub = el("button", "gh-hub");
      hub.type = "button";
      hub.setAttribute("aria-label", "Spin");
      hub.append(el("span", null, "转"), el("small", null, "Spin"));
      hub.addEventListener("click", go);
      cv.addEventListener("click", go);
      wrap.append(pointer, cv, hub);
      box.append(wrap);
      ctx = cv.getContext("2d");
      size = 0;
      winner = -1;
    }

    function setItems(l) {
      list = l;
      winner = -1;
      renderFace();
      draw(performance.now());
    }

    function resize(width) {
      // Leave room above for the readout and below for the buttons on a laptop screen.
      const s = Math.floor(Math.min(width, Math.max(260, Math.min(520, innerHeight - 390))));
      if (s === size && dpr === (devicePixelRatio || 1)) return;
      size = s;
      dpr = devicePixelRatio || 1;
      cv.style.width = cv.style.height = size + "px";
      cv.width = cv.height = Math.round(size * dpr);
      renderFace();
      draw(performance.now());
    }

    function renderFace() {
      if (!size) return;
      face = document.createElement("canvas");
      face.width = face.height = Math.round(size * dpr);
      const c = face.getContext("2d");
      c.scale(dpr, dpr);
      const R = size / 2;
      const rIn = R * 0.9;
      c.translate(R, R);
      c.fillStyle = css("--gh-rim");
      c.beginPath();
      c.arc(0, 0, R, 0, TAU);
      c.fill();

      const n = list.length;
      if (!n) {
        c.fillStyle = css("--line-strong");
        c.beginPath();
        c.arc(0, 0, rIn, 0, TAU);
        c.fill();
        return;
      }
      const a = TAU / n;
      const fills = [css("--gh-red"), css("--gh-gold"), css("--gh-jade")];
      const inks = [css("--gh-cream"), css("--gh-ink"), css("--gh-cream")];
      const tone = (i) => (n > 1 && n % 2 && i === n - 1 ? 2 : i % 2);
      for (let i = 0; i < n; i++) {
        const s = -Math.PI / 2 + i * a;
        c.fillStyle = fills[tone(i)];
        c.beginPath();
        c.moveTo(0, 0);
        c.arc(0, 0, rIn, s, s + a);
        c.closePath();
        c.fill();
      }
      if (n > 1) {
        c.strokeStyle = "rgba(0,0,0,0.18)";
        c.lineWidth = 1;
        for (let i = 0; i < n; i++) {
          const s = -Math.PI / 2 + i * a;
          c.beginPath();
          c.moveTo(0, 0);
          c.lineTo(rIn * Math.cos(s), rIn * Math.sin(s));
          c.stroke();
        }
      }

      // Radial labels with the emoji at the rim. Each label is sized so that its
      // innermost glyph still fits inside the slice and it stops short of the hub.
      const outer = rIn * 0.95;
      const hub = size * 0.105 + 8;
      const chord = 1.8 * Math.sin(Math.min(a, Math.PI) / 2);
      const family = font();
      c.font = "600 100px " + family;
      const fit = (t) => {
        const k = c.measureText(t).width / 100;
        return Math.min(26, (chord * outer) / (1 + chord * k), (outer - hub) / k);
      };
      // Each label keeps both emoji if it can stay 9 px tall, then one, then the
      // Chinese name alone, which is written down to 5 px.
      const labels = list.map((d) => {
        for (const t of [d.zh + " " + d.icon, d.zh + " " + d.emoji[0]]) {
          const fs = fit(t);
          if (fs >= 9) return { t, fs };
        }
        return { t: d.zh, fs: fit(d.zh) };
      });
      const typical = labels.map((l) => l.fs).sort((x, y) => x - y)[Math.floor(n / 2)];
      c.textAlign = "right";
      c.textBaseline = "middle";
      for (let i = 0; i < n; i++) {
        const fs = Math.min(labels[i].fs, typical);
        if (fs < 5) continue;
        c.save();
        c.rotate(-Math.PI / 2 + (i + 0.5) * a);
        c.fillStyle = inks[tone(i)];
        c.font = "600 " + fs.toFixed(1) + "px " + family;
        c.fillText(labels[i].t, outer, 0);
        c.restore();
      }
    }

    function draw(now) {
      if (!face) return;
      const R = size / 2;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, size, size);
      ctx.save();
      ctx.translate(R, R);
      ctx.rotate(theta);
      ctx.drawImage(face, -R, -R, size, size);
      if (winner >= 0) {
        const a = TAU / list.length;
        const t = (now - winAt) / 1000;
        ctx.fillStyle = "rgba(255,255,255," + (t < 2.4 ? 0.2 + 0.15 * Math.sin(t * 9) : 0.15) + ")";
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.arc(0, 0, R * 0.9, -Math.PI / 2 + winner * a, -Math.PI / 2 + (winner + 1) * a);
        ctx.closePath();
        ctx.fill();
      }
      // Motion blur: faint copies trailing the face while it moves fast.
      if (Math.abs(omega) > 0.03) {
        ctx.globalAlpha = 0.28;
        for (let i = 0; i < 2; i++) {
          ctx.rotate(-omega * 0.35);
          ctx.drawImage(face, -R, -R, size, size);
        }
        ctx.globalAlpha = 1;
      }
      ctx.restore();

      // Rim bulbs: steady when idle, chasing while spinning or celebrating.
      const bulbs = 28;
      const lit = css("--gh-light");
      const chase = !reduce.matches && (spinning || (winner >= 0 && now - winAt < 2400));
      const phase = Math.floor(now / (spinning ? 70 : 140));
      ctx.save();
      ctx.translate(R, R);
      ctx.fillStyle = lit;
      for (let i = 0; i < bulbs; i++) {
        const ang = (i / bulbs) * TAU;
        const bright = !chase || (i + phase) % 3 === 0;
        ctx.globalAlpha = bright ? 1 : 0.3;
        ctx.shadowColor = bright ? lit : "transparent";
        ctx.shadowBlur = bright ? R * 0.03 : 0;
        ctx.beginPath();
        ctx.arc(R * 0.95 * Math.cos(ang), R * 0.95 * Math.sin(ang), R * 0.017, 0, TAU);
        ctx.fill();
      }
      ctx.restore();
      pointer.style.transform = "rotate(" + (-32 * kick).toFixed(2) + "deg)";
    }

    function run(target, show) {
      const n = list.length;
      const land = -(target + rand(0.15, 0.85)) * (TAU / n);
      const turns = reduce.matches ? 1 : 4 + Math.floor(Math.random() * 3);
      const from = theta;
      const delta = turns * TAU + mod(land - theta, TAU);
      const dur = reduce.matches ? 1200 : rand(4200, 5600);
      const me = ++anim;
      let current = -1;
      let prev = theta;
      winner = -1;
      spinning = true;
      return frames((t, now) => {
        const p = Math.min(1, t / dur);
        theta = from + delta * (1 - Math.pow(1 - p, 4));
        omega = theta - prev;
        prev = theta;
        const i = indexAt(theta);
        if (i !== current) {
          current = i;
          kick = 1;
          sfx.tick();
          show(list[i]);
        }
        kick *= 0.82;
        draw(now);
        if (p < 1) return true;
        spinning = false;
        omega = 0;
        winner = target;
        winAt = now;
        // Let the pointer settle and the winning slice pulse for a while.
        frames((t2, now2) => {
          if (me !== anim || !cv.isConnected) return false;
          kick *= 0.82;
          draw(now2);
          return t2 < 2600;
        });
        return false;
      });
    }

    const retheme = () => { renderFace(); draw(performance.now()); };
    return { mount, setItems, resize, retheme, run };
  }

  // --- randomiser 2: the slot machine -----------------------------------------

  // Three reels: the dish's first emoji, its second, and its price. Each row
  // across the reels is one dish, and the reels stop left to right.
  function makeSlots() {
    let slot, lever;
    let reels = []; // {cv, ctx, pos, prev, speed}; pos is in rows, the payline shows list[round(pos) mod n]
    let list = [];
    let colW = 0;
    let rowH = 64;
    let dpr = 1;
    const symbol = (d, k) => (k === 2 ? "£" + d.price.toFixed(2) : d.emoji[k] || d.emoji[0]);

    function mount(box) {
      const wrap = el("div", "gh-slotbox");
      slot = el("div", "gh-slot");
      const marquee = el("div", "gh-slot-marquee");
      const name = el("span", "gh-slot-name", "回味轩");
      name.lang = "zh";
      marquee.append(name, el("span", "gh-slot-sub", "GOLDEN HOUSE"));
      const win = el("div", "gh-slot-window");
      const strip = el("div", "gh-slot-reels");
      reels = [0, 1, 2].map(() => {
        const cv = el("canvas", "gh-slot-reel");
        strip.append(cv);
        return { cv, ctx: cv.getContext("2d"), pos: 0, prev: 0, speed: 0 };
      });
      win.append(strip, el("div", "gh-slot-glass"), el("div", "gh-slot-band"));
      slot.append(marquee, win, el("div", "gh-slot-tray"));
      lever = el("button", "gh-lever");
      lever.type = "button";
      lever.setAttribute("aria-label", "Pull the lever");
      lever.append(el("span", "gh-lever-hub"), el("span", "gh-lever-arm"));
      lever.addEventListener("click", go);
      win.addEventListener("click", go);
      wrap.append(slot, lever);
      box.append(wrap);
      colW = 0;
    }

    function setItems(l) {
      list = l;
      reels.forEach((r) => (r.pos = 0));
      draw();
    }

    function resize(width) {
      // Cabinet at most 400 px with the lever beside it; 68 px of the cabinet is frame, bezel and reel gaps.
      const cw = Math.floor((Math.min(width - 48, 400) - 68) / 3);
      if (cw === colW && dpr === (devicePixelRatio || 1)) return;
      colW = cw;
      dpr = devicePixelRatio || 1;
      rowH = Math.round(Math.max(54, Math.min(76, colW * 0.62)));
      for (const r of reels) {
        r.cv.style.width = colW + "px";
        r.cv.style.height = rowH * 3 + "px";
        r.cv.width = Math.round(colW * dpr);
        r.cv.height = Math.round(rowH * 3 * dpr);
      }
      draw();
    }

    function drawReel(r, k) {
      const c = r.ctx;
      const h = rowH * 3;
      const n = list.length;
      c.setTransform(dpr, 0, 0, dpr, 0, 0);
      c.globalAlpha = 1;
      c.filter = "none";
      c.fillStyle = css("--gh-reel");
      c.fillRect(0, 0, colW, h);
      if (n) {
        const family = font();
        // Motion blur: smear the symbols over the distance moved this frame, and
        // soften them where the browser supports canvas filters.
        const samples = Math.max(1, Math.min(14, Math.ceil((Math.abs(r.speed) * rowH) / 2.5)));
        c.globalAlpha = Math.min(1, 2.2 / samples);
        if (samples > 1) c.filter = "blur(" + Math.min(2.5, (Math.abs(r.speed) * rowH) / 10).toFixed(1) + "px)";
        c.textAlign = "center";
        c.textBaseline = "middle";
        if (k === 2) {
          c.fillStyle = "#7a1d14";
          c.font = "800 " + Math.min(rowH * 0.32, colW * 0.22).toFixed(1) + "px " + family;
        } else {
          c.fillStyle = "#000"; // an emoji takes the alpha of the fill colour
          c.font = (rowH * 0.56).toFixed(1) + "px " + family;
        }
        for (let s = 0; s < samples; s++) {
          const p = r.pos - (r.speed * s) / samples;
          for (let j = Math.floor(p) - 2; j <= Math.floor(p) + 2; j++) {
            c.fillText(symbol(list[mod(j, n)], k), colW / 2, rowH * 1.5 + (j - p) * rowH);
          }
        }
        c.filter = "none";
        c.globalAlpha = 1;
      }
      // Shade top and bottom so each reel reads as a drum.
      const g = c.createLinearGradient(0, 0, 0, h);
      g.addColorStop(0, "rgba(0,0,0,0.5)");
      g.addColorStop(0.3, "rgba(0,0,0,0)");
      g.addColorStop(0.7, "rgba(0,0,0,0)");
      g.addColorStop(1, "rgba(0,0,0,0.5)");
      c.fillStyle = g;
      c.fillRect(0, 0, colW, h);
    }

    const draw = () => colW && reels.forEach(drawReel);

    function run(target, show) {
      const n = list.length;
      lever.classList.remove("is-pulled");
      void lever.offsetWidth;
      lever.classList.add("is-pulled");
      slot.classList.remove("is-win");
      slot.classList.add("is-running");
      sfx.clunk();
      // Overshoot by less than half a row, so the payline never shows the wrong dish.
      const over = reduce.matches ? 0 : 0.4;
      const delay = reduce.matches ? 0 : 220;
      const plans = reels.map((r, k) => {
        const start = Math.round(r.pos);
        const minRows = reduce.matches ? 3 + k : Math.round(rand(22, 30)) + k * 10;
        const offset = mod(target - start, n);
        r.pos = r.prev = start;
        return {
          r, start,
          final: start + offset + n * Math.ceil(Math.max(0, minRows - offset) / n),
          dur: reduce.matches ? 600 + k * 250 : rand(1500, 1800) + k * rand(550, 750),
          done: false,
        };
      });
      let current = -1;
      return frames((t) => {
        t -= delay;
        if (t < 0) return true;
        for (const pl of plans) {
          if (pl.done) continue;
          const { r, start, final, dur } = pl;
          const main = dur * 0.86;
          if (t < main) {
            r.pos = start + (final + over - start) * (1 - Math.pow(1 - t / main, 3));
          } else {
            const q = Math.min(1, (t - main) / (dur - main));
            r.pos = final + over * (1 - q * q * (3 - 2 * q));
          }
          if (t >= dur) {
            r.pos = final;
            pl.done = true;
            sfx.clunk();
          }
          r.speed = pl.done ? 0 : r.pos - r.prev;
          r.prev = r.pos;
        }
        // The readout follows the last reel, the one that settles the dish.
        const i = mod(Math.round(reels[2].pos), n);
        if (i !== current) {
          current = i;
          sfx.tick();
          show(list[i]);
        }
        draw();
        if (plans.some((pl) => !pl.done)) return true;
        slot.classList.remove("is-running");
        slot.classList.add("is-win");
        return false;
      });
    }

    return { mount, setItems, resize, retheme: draw, run };
  }

  // --- randomiser 3: fortune sticks (求签) ------------------------------------

  const STICK_RISE = 156; // px from where the sticks stand to the cup mouth; see .gh-cup in picker.css

  function makeSticks() {
    let cup, out;
    let list = [];

    function mount(box) {
      const wrap = el("div", "gh-sticks");
      cup = el("button", "gh-cup");
      cup.type = "button";
      cup.setAttribute("aria-label", "Shake the cup");
      const bundle = el("span", "gh-cup-sticks");
      // Each stick stands on the cup floor and leans so that it crosses the
      // mouth inside the rim, even at the far end of its jiggle.
      for (let i = 0; i < 15; i++) {
        const mouth = -40 + (80 * (i + Math.random())) / 15;
        const foot = mouth * 0.4 + rand(-8, 8);
        const s = el("i");
        s.style.setProperty("--x", foot.toFixed(1) + "px");
        s.style.setProperty("--r", ((Math.atan2(mouth - foot, STICK_RISE) * 180) / Math.PI).toFixed(1) + "deg");
        s.style.setProperty("--len", Math.round(rand(250, 290)) + "px");
        s.style.setProperty("--d", rand(0, 0.13).toFixed(2) + "s");
        bundle.append(s);
      }
      const body = el("span", "gh-cup-body");
      body.append(el("span", "gh-cup-label", "灵签"));
      cup.append(el("span", "gh-cup-mouth"), bundle, body);
      cup.addEventListener("click", go);
      out = el("div", "gh-stick-out");
      wrap.append(cup, out);
      box.append(wrap);
    }

    async function run(target) {
      out.classList.remove("is-out");
      cup.classList.add("is-shaking");
      const dur = reduce.matches ? 400 : rand(1700, 2300);
      let next = 0;
      await frames((t) => {
        if (t >= next) {
          sfx.clack();
          next = t + rand(55, 130);
        }
        return t < dur;
      });
      cup.classList.remove("is-shaking");
      const d = list[target];
      const dish = el("span", "gh-stick-dish", d.zh);
      dish.style.fontSize = Math.min(26, 200 / d.zh.length).toFixed(1) + "px";
      out.replaceChildren(
        el("span", "gh-stick-no", "第" + zhNum(target + 1) + "签"),
        el("span", "gh-stick-emoji", d.emoji.join("\n")),
        dish,
        el("span", "gh-stick-luck", "上上签"),
      );
      void out.offsetWidth;
      out.classList.add("is-out");
      sfx.pop();
      await wait(reduce.matches ? 0 : 850);
    }

    return { mount, setItems: (l) => (list = l), resize() {}, retheme() {}, run };
  }

  // --- the page -------------------------------------------------------------

  const MODES = {
    wheel: { label: "Wheel", verb: "转", action: "Spin", prompt: "Spin the wheel, or press space", ...makeWheel() },
    slots: { label: "Slots", verb: "拉", action: "Pull", prompt: "Pull the lever, or press space", ...makeSlots() },
    sticks: {
      label: "Sticks", verb: "摇", action: "Shake", prompt: "Shake the fortune sticks, or press space",
      waiting: ["", "摇签中…", "Shaking the sticks…"], ...makeSticks(),
    },
  };
  const STATES = { 0: "either", 1: "must have", "-1": "leave out" };

  const dialog = $(".gh-menu");
  const sections = [];       // {group, cat, chip}
  const on = new Set();      // indices into sections
  const skipped = new Set(); // dish ids taken out by hand
  const rows = new Map();    // dish id -> {d, row, btn, text} in the menu
  const history = [];        // recent picks, newest first
  let attrs = [];            // {id, name, emoji, kind, tags?, chip, state}; state 1 = must have, -1 = leave out
  let tagText = {};
  let prices = [];           // distinct menu prices, for the budget slider
  let budget = Infinity;
  let all = [];              // every dish on the menu
  let items = [];            // dishes in the draw
  let winner = null;
  let mode = null;
  let busy = false;
  let lastWidth = 0;

  const hasAttr = (a, d) => (a.tags ? a.tags.some((t) => (d.tags || []).includes(t)) : (d.has || []).includes(a.id));
  const passes = (d, except) => attrs.every((a) => a === except || !a.state || hasAttr(a, d) === a.state > 0);
  // In the draw unless its section, an attribute, the budget or a skip rules it out.
  const eligible = (d, except) => on.has(d.section) && d.price <= budget && passes(d, except);
  const inDraw = (d) => eligible(d) && !skipped.has(d.id);

  function setReadout(emoji, zh, en, meta) {
    const z = $(".gh-r-zh");
    z.replaceChildren();
    if (emoji) z.append(el("span", "gh-r-emoji", emoji));
    z.append(zh);
    $(".gh-r-en").textContent = en;
    $(".gh-r-meta").textContent = meta;
  }

  function showDish(d) {
    const tags = (d.tags || []).map((t) => tagText[t]);
    const meta = [d.where, d.region, "£" + d.price.toFixed(2), ...tags, d.extra].filter(Boolean);
    setReadout(d.icon, d.zh, d.en + (d.note ? " (" + d.note + ")" : ""), meta.join(" · "));
  }

  function idle() {
    readout.classList.remove("is-win");
    stage.classList.remove("is-win");
    $(".gh-after").classList.remove("is-on");
    const hint = !on.size ? "Pick at least one section" : "Nothing matches. Loosen a filter or the budget.";
    setReadout("", "今天吃什么？", mode.prompt, items.length ? "" : hint);
  }

  // --- the draw panel ---------------------------------------------------------

  function buildSections(groups) {
    const box = $(".gh-sections");
    groups.forEach((g, gi) => {
      const card = el("div", "gh-box");
      const head = el("div", "gh-box-head");
      const title = el("span", null, g.emoji + " " + g.name + " ");
      title.append(el("span", "gh-zh", g.zh));
      const links = el("span", "gh-links");
      const allBtn = el("button", "gh-link", "All");
      const noneBtn = el("button", "gh-link", "None");
      allBtn.type = noneBtn.type = "button";
      links.append(allBtn, noneBtn);
      head.append(title, links);
      const wrap = el("div", "gh-chips");
      const mine = [];
      g.cats.forEach((c) => {
        const i = sections.length;
        const chip = el("button", "gh-chip", c.emoji + " " + (g.cats.length === 1 ? "All " + g.name.toLowerCase() : c.name));
        chip.type = "button";
        chip.append(el("small", null, String(c.items.length)));
        chip.addEventListener("click", () => setSections([i], !on.has(i)));
        sections.push({ group: g, cat: c, chip });
        mine.push(i);
        wrap.append(chip);
        if (gi === 0) on.add(i); // default: the special rice bowls
      });
      allBtn.addEventListener("click", () => setSections(mine, true));
      noneBtn.addEventListener("click", () => setSections(mine, false));
      if (g.cats.length === 1) links.hidden = true;
      card.append(head, wrap);
      box.append(card);
    });
  }

  function buildAttrs(list) {
    attrs = list.map((a) => {
      const chip = el("button", "gh-chip gh-tri", a.emoji + " " + a.name);
      chip.type = "button";
      chip.append(el("small"));
      const attr = { ...a, chip, state: 0 };
      chip.addEventListener("click", () => {
        if (busy) return;
        attr.state = attr.state === 0 ? 1 : attr.state === 1 ? -1 : 0;
        refresh();
      });
      $(a.kind === "ingredient" ? ".gh-attr-ingredient" : ".gh-attr-heat").append(chip);
      return attr;
    });
  }

  // The budget slider steps through the prices in the selected sections; its top step means any price.
  function syncBudget() {
    prices = [...new Set(all.filter((d) => on.has(d.section)).map((d) => d.price))].sort((a, b) => a - b);
    const range = $(".gh-budget-range");
    range.max = prices.length;
    if (budget === Infinity || !prices.length) {
      range.value = prices.length;
    } else {
      const i = Math.max(0, prices.findLastIndex((p) => p <= budget));
      budget = prices[i];
      range.value = i;
    }
    range.disabled = busy || !prices.length;
  }

  function buildBudget() {
    const range = $(".gh-budget-range");
    range.min = 0;
    range.step = 1;
    range.addEventListener("input", () => {
      budget = +range.value >= prices.length ? Infinity : prices[+range.value];
      refresh();
    });
  }

  function setSections(idx, state) {
    if (busy) return;
    idx.forEach((i) => (state ? on.add(i) : on.delete(i)));
    refresh();
  }

  function reset() {
    if (busy) return;
    on.clear();
    sections.forEach((s, i) => s.group === sections[0].group && on.add(i));
    attrs.forEach((a) => (a.state = 0));
    skipped.clear();
    budget = Infinity;
    refresh();
  }

  function refresh() {
    syncBudget();
    items = all.filter(inDraw);
    for (let i = items.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [items[i], items[j]] = [items[j], items[i]];
    }
    sections.forEach((s, i) => s.chip.setAttribute("aria-pressed", on.has(i)));
    // Each attribute chip counts the dishes that have it among what the other settings let through.
    for (const a of attrs) {
      const n = all.filter((d) => eligible(d, a) && !skipped.has(d.id) && hasAttr(a, d)).length;
      a.chip.lastChild.textContent = n;
      a.chip.dataset.state = a.state;
      a.chip.setAttribute("aria-label", a.name + ", " + n + " dishes: " + STATES[a.state]);
      a.chip.classList.toggle("is-none", !n && !a.state);
    }
    $(".gh-budget-out").textContent = budget === Infinity ? "Any price" : "Up to £" + budget.toFixed(2);
    $$(".gh-count").forEach((c) => (c.textContent = items.length + (items.length === 1 ? " dish" : " dishes") + " in the draw"));
    const unskip = $(".gh-unskip");
    unskip.hidden = !skipped.size;
    unskip.textContent = "Put back " + skipped.size + " skipped";
    goBtn.disabled = busy || !items.length;
    mode.setItems(items);
    if (dialog.open) filterMenu();
    idle();
  }

  // --- the menu -----------------------------------------------------------------

  function buildMenu(groups) {
    const list = $(".gh-menu-list");
    const ingredient = Object.fromEntries(attrs.map((a) => [a.id, a.name]));
    for (const g of groups) {
      const box = el("section", "gh-mgroup");
      const head = el("div", "gh-mgroup-head", g.emoji + " " + g.name + " ");
      head.append(el("span", "gh-zh", g.zh));
      box.append(head);
      if (g.note) box.append(el("div", "gh-mnote", g.note));
      for (const c of g.cats) {
        const sec = el("div", "gh-mcat");
        if (g.cats.length > 1) sec.append(el("div", "gh-mcat-head", c.emoji + " " + c.name));
        if (c.note) sec.append(el("div", "gh-mnote", c.note));
        for (const d of c.items) {
          const tags = (d.tags || []).map((t) => tagText[t]);
          const name = el("div", "gh-dish-name");
          name.append(el("span", "gh-dish-zh", d.zh));
          name.append(el("span", "gh-dish-en", [d.en + (d.note ? " (" + d.note + ")" : ""), d.region].filter(Boolean).join(" · ")));
          if (tags.length) name.append(el("span", "gh-dish-tags", tags.join("   ")));
          const btn = el("button", "gh-row-btn");
          btn.type = "button";
          btn.addEventListener("click", () => {
            if (busy) return;
            if (skipped.has(d.id)) skipped.delete(d.id);
            else skipped.add(d.id);
            refresh();
          });
          const row = el("div", "gh-dish");
          row.append(el("span", "gh-dish-emoji", d.icon), name, el("span", "gh-dish-price", "£" + d.price.toFixed(2)), btn);
          sec.append(row);
          // Section names stay out of the search text: "Vegetables & Tofu" would match every dish in it.
          const words = [d.zh, d.en, d.note, d.region, ...tags, ...(d.has || []).map((h) => ingredient[h])];
          rows.set(d.id, { d, row, btn, text: words.filter(Boolean).join(" ").toLowerCase() });
        }
        box.append(sec);
      }
      list.append(box);
    }
    list.append(el("div", "gh-mnote gh-menu-empty", "No dishes match."));
  }

  // Search matches every word typed, in English, Chinese, tags or ingredients.
  function filterMenu() {
    const words = $(".gh-search").value.trim().toLowerCase().split(/\s+/).filter(Boolean);
    const only = $(".gh-only").checked;
    let shown = 0;
    for (const r of rows.values()) {
      const drawn = inDraw(r.d);
      const skip = skipped.has(r.d.id);
      // A skipped dish stays listed with the draw, so it can be put back.
      const show = words.every((w) => r.text.includes(w)) && (!only || drawn || (skip && eligible(r.d)));
      r.row.hidden = !show;
      shown += show;
      r.row.classList.toggle("is-out", !drawn);
      r.row.classList.toggle("is-picked", !!winner && winner.id === r.d.id);
      r.btn.hidden = !drawn && !(skip && eligible(r.d));
      r.btn.textContent = skip ? "Put back" : "Skip";
    }
    $$(".gh-mcat").forEach((sec) => (sec.hidden = !sec.querySelector(".gh-dish:not([hidden])")));
    $$(".gh-mgroup").forEach((box) => (box.hidden = !box.querySelector(".gh-dish:not([hidden])")));
    $(".gh-menu-empty").hidden = shown > 0;
    const scope = only ? items.length + " in the draw" : all.length + " on the menu";
    $(".gh-menu-count").textContent = words.length ? shown + " match · " + scope : scope;
  }

  function openMenu(focusId) {
    if (focusId != null) {
      $(".gh-search").value = "";
      if (!items.some((d) => d.id === focusId)) $(".gh-only").checked = false;
    }
    filterMenu();
    if (!dialog.open) dialog.showModal();
    const r = rows.get(focusId);
    if (r) {
      r.row.scrollIntoView({ block: "center" });
      r.row.classList.remove("is-flash");
      void r.row.offsetWidth;
      r.row.classList.add("is-flash");
    }
  }

  // --- picking ----------------------------------------------------------------------

  function buildModes() {
    const box = $(".gh-modes");
    for (const [key, m] of Object.entries(MODES)) {
      const b = el("button", "gh-mode", m.label);
      b.type = "button";
      b.dataset.mode = key;
      b.addEventListener("click", () => setMode(key));
      box.append(b);
    }
  }

  function setMode(key) {
    if (busy || !MODES[key]) return;
    mode = MODES[key];
    store.set("gh-mode", key);
    $$(".gh-mode").forEach((b) => b.setAttribute("aria-pressed", b.dataset.mode === key));
    stage.className = "gh-stage";
    stage.replaceChildren();
    mode.mount(stage);
    mode.setItems(items);
    lastWidth = stage.clientWidth;
    mode.resize(lastWidth);
    const verb = el("span", null, mode.verb);
    verb.lang = "zh";
    goBtn.replaceChildren(verb, mode.action + "!");
    idle();
  }

  function lock(state) {
    busy = state;
    root.classList.toggle("is-busy", state);
    $$(".gh-mode, .gh-chip, .gh-panel .gh-link, .gh-budget-range, .gh-after button, .gh-row-btn").forEach((b) => (b.disabled = state));
    goBtn.disabled = state || !items.length;
  }

  function remember(d) {
    history.unshift(d);
    if (history.length > 5) history.pop();
    const box = $(".gh-history");
    box.replaceChildren(el("div", "gh-history-label", "Recent picks"));
    for (const h of history) {
      const b = el("button", "gh-hist");
      b.type = "button";
      b.title = "Show on the menu";
      const name = el("span", "gh-hist-name");
      name.append(el("span", "gh-hist-zh", h.zh), el("span", "gh-hist-en", h.en));
      b.append(el("span", "gh-hist-emoji", h.icon), name, el("span", "gh-hist-price", "£" + h.price.toFixed(2)));
      b.addEventListener("click", () => openMenu(h.id));
      box.append(b);
    }
  }

  async function go() {
    if (busy || !items.length) return;
    sfx.wake();
    lock(true);
    const target = Math.floor(Math.random() * items.length);
    readout.classList.remove("is-win");
    stage.classList.remove("is-win");
    $(".gh-after").classList.remove("is-on");
    $(".gh-status").textContent = "Picking…";
    if (mode.waiting) setReadout(...mode.waiting, "");
    // Bring the readout and stage into view if the pick started off-screen.
    const top = readout.getBoundingClientRect().top;
    if (top < 0 || top > innerHeight * 0.4) readout.scrollIntoView({ behavior: reduce.matches ? "auto" : "smooth" });

    await mode.run(target, showDish);

    lock(false);
    const d = items[target];
    winner = d;
    showDish(d);
    void readout.offsetWidth; // restart the pop animation
    readout.classList.add("is-win");
    stage.classList.add("is-win");
    $(".gh-after").classList.add("is-on");
    $(".gh-status").textContent = "You got " + d.en + ", " + d.zh + ", £" + d.price.toFixed(2);
    remember(d);
    sfx.fanfare();
    const box = stage.getBoundingClientRect();
    burst(box.left + box.width / 2, box.top + box.height / 2, { glyphs: Array.from({ length: 12 }, (_, i) => d.emoji[i % d.emoji.length]) });
  }

  function showSound() {
    soundBtn.setAttribute("aria-pressed", sfx.on);
    soundBtn.setAttribute("aria-label", sfx.on ? "Sound on" : "Sound off");
    soundBtn.replaceChildren(sfx.on ? "🔊" : "🔇", el("span", "gh-sound-label", sfx.on ? " Sound on" : " Sound off"));
  }

  // --- wiring -------------------------------------------------------------------------

  goBtn.addEventListener("click", go);
  soundBtn.addEventListener("click", () => {
    sfx.toggle();
    showSound();
  });
  $(".gh-skip-winner").addEventListener("click", () => {
    if (busy || !winner) return;
    skipped.add(winner.id);
    refresh();
    go();
  });
  $(".gh-find-winner").addEventListener("click", () => winner && openMenu(winner.id));
  $$(".gh-open-menu").forEach((b) => b.addEventListener("click", () => openMenu()));
  $(".gh-jump").addEventListener("click", () => $(".gh-panel").scrollIntoView({ behavior: reduce.matches ? "auto" : "smooth" }));
  $(".gh-menu-close").addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", (e) => e.target === dialog && dialog.close()); // click on the backdrop
  $(".gh-search").addEventListener("input", filterMenu);
  $(".gh-only").addEventListener("change", filterMenu);
  $(".gh-attrs-clear").addEventListener("click", () => {
    if (busy) return;
    attrs.forEach((a) => (a.state = 0));
    refresh();
  });
  $(".gh-unskip").addEventListener("click", () => {
    if (busy) return;
    skipped.clear();
    refresh();
  });
  $(".gh-reset").addEventListener("click", reset);
  document.addEventListener("keydown", (e) => {
    if (e.code !== "Space" || dialog.open || e.target.closest("button, input, textarea, select, a")) return;
    e.preventDefault();
    go();
  });
  new ResizeObserver(() => {
    if (!mode || stage.clientWidth === lastWidth) return;
    lastWidth = stage.clientWidth;
    mode.resize(lastWidth);
  }).observe(root);
  addEventListener("resize", () => mode && mode.resize(stage.clientWidth));
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => mode && mode.retheme());

  showSound();
  fetch(menuUrl)
    .then((r) => r.json())
    .then((m) => {
      tagText = m.tagText;
      let section = 0;
      for (const g of m.groups) {
        for (const c of g.cats) {
          const where = g.cats.length === 1 ? g.name : g.name + " · " + c.name;
          for (const d of c.items) {
            all.push(Object.assign(d, { id: all.length, section, where, extra: c.note || g.note, icon: d.emoji.join("") }));
          }
          section++;
        }
      }
      buildSections(m.groups);
      buildAttrs(m.attributes);
      buildBudget();
      buildMenu(m.groups);
      buildModes();
      setMode(MODES[store.get("gh-mode", "wheel")] ? store.get("gh-mode", "wheel") : "wheel");
      refresh();
    })
    .catch(() => {
      $(".gh-r-en").textContent = "Could not load the menu.";
    });
})();
