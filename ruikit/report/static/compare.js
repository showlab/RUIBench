
const D = JSON.parse(document.getElementById('data').textContent);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const short = p => D.names[p][0];
const clock = t => t === '0' ? '思考不计时' : `每次回复 ${t} s`;
const shade = k => D.lat.length === 4 ? `var(--lat4-${k})` : `var(--lat-${Math.min(k, 2)})`;
const ICON = {ok: '✓', timeout: '⏱', budget: '⊘', failed: '✗', error: '!', none: '·'};
const WORD = {ok: '成功', timeout: '超时', budget: '回复或决策轮用完', failed: '没做成', error: '运行出错'};
const NTASK = Object.keys(D.tasks).length;

// -- tooltip ------------------------------------------------------------------
const tip = document.createElement('div'); tip.className = 'tip'; document.body.appendChild(tip);
function showTip(ev, htmlText) {
  tip.innerHTML = htmlText; tip.style.display = 'block';
  const r = tip.getBoundingClientRect();
  let x = ev.clientX + 14, y = ev.clientY + 14;
  if (x + r.width > innerWidth - 8) x = ev.clientX - r.width - 14;
  if (y + r.height > innerHeight - 8) y = ev.clientY - r.height - 14;
  tip.style.left = Math.max(8, x) + 'px'; tip.style.top = Math.max(8, y) + 'px';
}
function hideTip() { tip.style.display = 'none'; }
function hover(el, text) {
  el.addEventListener('pointermove', ev => showTip(ev, text()));
  el.addEventListener('pointerleave', hideTip);
  el.addEventListener('focus', ev => { const b = el.getBoundingClientRect(); showTip({clientX: b.right, clientY: b.bottom}, text()); });
  el.addEventListener('blur', hideTip);
}

// -- 1. success by interface and clock: grouped columns ------------------------
function successChart() {
  const host = document.getElementById('success');
  const W = Math.max(320, host.clientWidth), H = 280;
  const m = {l: 34, r: 8, t: 22, b: 40};
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const series = (Object.keys(D.oracle).length ? ['ref'] : []).concat(D.lat);
  const nL = series.length, band = iw / D.profiles.length;
  const bw = Math.min(24, (band * 0.7 - 2 * (nL - 1)) / nL);
  const y = v => m.t + ih - ih * v / NTASK;
  let s = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="各接口在不同思考时间下完成的任务数">`;
  const step = NTASK > 12 ? 6 : 2;
  for (let v = 0; v <= NTASK; v += step) {
    s += `<line class="${v ? 'grid' : 'base'}" x1="${m.l}" x2="${W - m.r}" y1="${y(v)}" y2="${y(v)}"/>`;
    s += `<text x="${m.l - 6}" y="${y(v) + 4}" text-anchor="end">${v}</text>`;
  }
  const hits = [];
  D.profiles.forEach((p, i) => {
    const gx = m.l + band * i + band / 2, w = nL * bw + (nL - 1) * 2;
    series.forEach((t, k) => {
      const sm = t === 'ref' ? D.oracle[p] : D.summary[`${p}|${t}`]; if (!sm) return;
      const x = gx - w / 2 + k * (bw + 2), top = y(sm.ok), h = m.t + ih - top;
      const r = Math.min(4, h, bw / 2);
      const path = h > 0 ? `M${x},${m.t + ih} V${top + r} Q${x},${top} ${x + r},${top} H${x + bw - r} Q${x + bw},${top} ${x + bw},${top + r} V${m.t + ih} Z` : '';
      const fill = t === 'ref' ? 'var(--ref)' : shade(k - (series.length - D.lat.length));
      s += `<path d="${path}" fill="${fill}" data-k="${p}|${t}"/>`;
      if (bw >= 16) s += `<text class="val" x="${x + bw / 2}" y="${top - 5}" text-anchor="middle">${sm.ok}</text>`;
      hits.push([x - 1, m.t, bw + 2, ih, p, t]);
    });
    s += `<text class="grp" x="${gx}" y="${H - 16}" text-anchor="middle">${esc(short(p))}</text>`;
  });
  for (const [x, yy, w, h, p, t] of hits) s += `<rect class="hit" x="${x}" y="${yy}" width="${w}" height="${h}" data-p="${p}" data-t="${t}"/>`;
  s += '</svg>';
  host.innerHTML = s;
  host.querySelectorAll('rect.hit').forEach(el => hover(el, () => {
    const p = el.dataset.p, t = el.dataset.t;
    if (t === 'ref') {
      const o = D.oracle[p];
      return `<b>${esc(short(p))}</b> · 用仿真真值坐标<br>完成 <b>${o.ok}</b> / ${o.n} 个任务<br><span class="m">思考不计时；只看工具够不够把事做出来</span>`;
    }
    const sm = D.summary[`${p}|${t}`];
    const rest = ['timeout', 'budget', 'failed', 'error'].filter(k => sm[k]).map(k => `${WORD[k]} ${sm[k]}`).join('，');
    return `<b>${esc(short(p))}</b> · ${clock(t)}<br>完成 <b>${sm.ok}</b> / ${sm.n} 个任务${rest ? `<br><span class="m">${rest}</span>` : ''}<br><span class="m">回复次数中位数 ${sm.med_replies}</span>`;
  }));
}
function successTable() {
  const ref = Object.keys(D.oracle).length;
  let h = '<table class="data"><thead><tr><th>接口</th>' + (ref ? '<th>用真值坐标</th>' : '') + D.lat.map(t => `<th>${clock(t)}</th>`).join('') + `<th>回复次数中位数（${clock(D.lat[0])}）</th></tr></thead><tbody>`;
  for (const p of D.profiles) {
    const o = D.oracle[p];
    h += `<tr><td>${esc(short(p))}</td>` + (ref ? `<td class="num">${o ? `${o.ok}/${o.n}` : '—'}</td>` : '') + D.lat.map(t => {
      const sm = D.summary[`${p}|${t}`];
      return `<td class="num">${sm ? `${sm.ok}/${sm.n}` : '—'}</td>`;
    }).join('') + `<td class="num">${(D.summary[`${p}|${D.lat[0]}`] || {}).med_replies ?? '—'}</td></tr>`;
  }
  return h + '</tbody></table>';
}

// -- 2. replies each task took, against the budget: a strip per interface -------
function repliesChart() {
  const host = document.getElementById('replies');
  const t0 = D.lat[0];
  const W = Math.max(320, host.clientWidth), rowH = 54, narrow = W < 640;
  const m = {l: Math.min(96, W * 0.22), r: 14, t: narrow ? 58 : 34, b: 30};
  const H = m.t + rowH * D.profiles.length + m.b, iw = W - m.l - m.r;
  const lo = Math.log10(0.03), hi = Math.log10(12);
  const x = v => m.l + iw * (Math.log10(Math.min(12, Math.max(0.03, v))) - lo) / (hi - lo);
  let s = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="每个任务用了多少次回复，与任务预算相比">`;
  for (const v of [0.03, 0.1, 0.3, 1, 3, 10]) {
    s += `<line class="grid" x1="${x(v)}" x2="${x(v)}" y1="${m.t}" y2="${H - m.b}"/>`;
    s += `<text x="${x(v)}" y="${H - m.b + 16}" text-anchor="middle">${v}×</text>`;
  }
  const lines = [[1, '8 s 一次：时间用完'], [4, '2 s 一次'], [10, '回复上限']];
  lines.forEach(([v, label], k) => {
    // one row of labels when they fit, else one row each (the last one right-aligned)
    const ly = narrow ? m.t - 10 - 14 * (2 - k) : m.t - 10;
    s += `<line class="limit" x1="${x(v)}" x2="${x(v)}" y1="${ly + 4}" y2="${H - m.b}"/>`;
    s += `<text class="limit-label" x="${x(v) + (k === 2 ? -4 : 4)}" y="${ly}" text-anchor="${k === 2 ? 'end' : 'start'}">${label}</text>`;
  });
  const ids = Object.keys(D.tasks);
  D.profiles.forEach((p, i) => {
    const cy = m.t + rowH * i + rowH / 2;
    s += `<text class="grp" x="${m.l - 10}" y="${cy + 4}" text-anchor="end">${esc(short(p))}</text>`;
    if (i) s += `<line class="grid" x1="${m.l}" x2="${W - m.r}" y1="${m.t + rowH * i}" y2="${m.t + rowH * i}"/>`;
    ids.forEach((id, j) => {
      const c = D.cells[`${id}|${p}|${t0}`]; if (!c) return;
      const v = c.replies / D.tasks[id].N;
      const jy = cy + ((j * 7) % 5 - 2) * 5;
      const fill = c.o === 'ok' ? 'var(--dot)' : 'var(--surface)';
      s += `<circle cx="${x(v)}" cy="${jy}" r="5" fill="${fill}" stroke="var(--dot)" stroke-width="2"/>`;
      s += `<circle class="hit" cx="${x(v)}" cy="${jy}" r="9" data-id="${id}" data-p="${p}"/>`;
    });
  });
  s += '</svg>';
  host.innerHTML = s;
  host.querySelectorAll('circle.hit').forEach(el => hover(el, () => {
    const id = el.dataset.id, p = el.dataset.p, T = D.tasks[id];
    const c = D.cells[`${id}|${p}|${t0}`];
    const others = D.lat.slice(1).map(t => { const o = D.cells[`${id}|${p}|${t}`]; return o ? `${clock(t)}：${o.why}` : ''; }).filter(Boolean).join('<br>');
    return `<b>${id} ${esc(T.name)}</b> · ${esc(short(p))}<br>${clock(t0)}：${c.replies} 次回复 = 预算 N=${T.N} 的 ${(c.replies / T.N).toFixed(2)} 倍；${c.why}${others ? `<br><span class="m">${others}</span>` : ''}`;
  }));
}
function repliesTable() {
  const t0 = D.lat[0];
  let h = '<table class="data"><thead><tr><th>任务</th><th>N</th>' + D.profiles.map(p => `<th>${esc(short(p))}</th>`).join('') + '</tr></thead><tbody>';
  for (const [id, T] of Object.entries(D.tasks)) {
    h += `<tr><td>${id}</td><td class="num">${T.N}</td>` + D.profiles.map(p => {
      const c = D.cells[`${id}|${p}|${t0}`];
      return `<td class="num">${c ? `${c.replies}（${(c.replies / T.N).toFixed(2)}×）${c.o === 'ok' ? '' : ' ✗'}` : '—'}</td>`;
    }).join('') + '</tr>';
  }
  return h + '</tbody></table>';
}

// -- 3. every task: the outcome under each clock ---------------------------------
function matrix() {
  const head = '<tr><th>任务 · 上限</th>' + D.profiles.map(p => `<th>${esc(short(p))}<br><span class="clocks">${D.lat.map(t => t + ' s').join(' · ')}</span></th>`).join('') + '</tr>';
  let rows = '';
  for (const s of D.suites) {
    rows += `<tr class="suite"><td colspan="${D.profiles.length + 1}">${esc(s.name)}</td></tr>`;
    for (const id of s.tasks) {
      const T = D.tasks[id];
      const clip = D.shows[D.lat[0]] ? `<a class="clip" href="${esc(D.shows[D.lat[0]])}#${id}">看录像 →</a>` : '';
      rows += `<tr><td><span class="id">${id}</span>${esc(T.name)}${clip}<span class="lim">N=${T.N} · ≤${T.replies} 次回复 · ≤${T.sim} s</span></td>`;
      for (const p of D.profiles) {
        rows += '<td><span class="chips">' + D.lat.map(t => {
          const c = D.cells[`${id}|${p}|${t}`];
          const o = c ? c.o : 'none';
          return `<span class="chip ${o}" tabindex="0" data-id="${id}" data-p="${p}" data-t="${t}" aria-label="${esc(short(p))} ${clock(t)} ${c ? esc(c.why) : '没跑'}">${ICON[o]}</span>`;
        }).join('') + '</span></td>';
      }
      rows += '</tr>';
    }
  }
  document.getElementById('matrix').innerHTML = `<div class="scroll"><table class="matrix"><thead>${head}</thead><tbody>${rows}</tbody></table></div>`;
  document.querySelectorAll('#matrix .chip').forEach(el => hover(el, () => {
    const {id, p, t} = el.dataset, c = D.cells[`${id}|${p}|${t}`], T = D.tasks[id];
    if (!c) return `<b>${id}</b> · ${esc(short(p))} · ${clock(t)}<br>没跑`;
    const time = t === '0' ? `仿真 ${c.sim} s（全是机器人动作）` : `仿真 ${c.sim} s，其中思考 ${c.think} s`;
    return `<b>${id} ${esc(T.name)}</b><br>${esc(short(p))} · ${clock(t)}<br>${ICON[c.o]} ${esc(c.why)}<br><span class="m">${c.replies} 次回复 / 上限 ${T.replies}；${time} / 上限 ${T.sim} s</span>`;
  }));
}

function toggles() {
  document.querySelectorAll('button.toggle').forEach(b => {
    const target = document.getElementById(b.dataset.for), table = document.getElementById(b.dataset.for + '-table');
    b.addEventListener('click', () => {
      const showTable = table.hidden;
      table.hidden = !showTable; target.hidden = showTable;
      b.textContent = showTable ? '看图' : '看表';
      b.setAttribute('aria-pressed', String(showTable));
    });
  });
  document.getElementById('success-table').innerHTML = successTable();
  document.getElementById('replies-table').innerHTML = repliesTable();
}

function draw() { successChart(); repliesChart(); }
matrix(); toggles(); draw();
let pending = 0;
addEventListener('resize', () => { cancelAnimationFrame(pending); pending = requestAnimationFrame(draw); });
