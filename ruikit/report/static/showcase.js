
const D = JSON.parse(document.getElementById('data').textContent);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function fmt(t) { return (Math.round(t * 10) / 10).toFixed(1); }
function overview() {
  let rows = '';
  for (const s of D.suites) {
    // the agents' page lists the tasks they have played; the full grid is the scripted page's
    const tasks = D.mode === 'agent' ? s.tasks.filter(t => D.clips[t.id]) : s.tasks;
    if (!tasks.length) continue;
    rows += `<tr class="suite"><td colspan="${D.profiles.length + 1}">${esc(s.name)}</td></tr>`;
    for (const t of tasks) {
      rows += `<tr><td><a href="#${t.id}"><span class="id">${t.id}</span>${esc(t.name)}</a></td>`;
      for (const p of D.profiles) {
        const c = (D.clips[t.id] || {})[p];
        rows += c ? `<td class="cell"><a href="#${t.id}"><span class="${c.success ? 'ok' : 'no'}">${c.success ? '✓' : '✗'}</span>`
                    + `<span class="n">${c.replies} 次决策 · ${fmt(c.sim_seconds)} s</span></a></td>`
                  : `<td class="cell"><span class="n">—</span></td>`;
      }
      rows += '</tr>';
    }
  }
  const head = '<tr><th>任务</th>' + D.profiles.map(p => `<th>${esc(D.names[p][0])}</th>`).join('') + '</tr>';
  // each interface over the tasks it was run on: successes, and the middle reply count
  const median = xs => { const v = [...xs].sort((a, b) => a - b), m = v.length >> 1;
                         return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2; };
  let total = '<tr class="total"><td>合计</td>';
  for (const p of D.profiles) {
    const cs = Object.values(D.clips).map(byP => byP[p]).filter(Boolean);
    // the world waited while the agent thought; charged 8 s a reply, would it still fit?
    const fits = cs.filter(c => c.success && c.budget && c.budget.max_sim_seconds
                           && c.sim_seconds + 8 * c.replies <= c.budget.max_sim_seconds).length;
    const won = cs.filter(c => c.success).length;
    const zeroClock = cs.every(c => !c.think_seconds);
    total += cs.length
      ? `<td class="cell"><span class="n"><b>${won}/${cs.length}</b> 成功（${Math.round(100 * won / cs.length)}%）· 中位 ${median(cs.map(c => c.replies))} 次决策`
        + (D.mode === 'agent' && zeroClock ? `<br>按每次 8 s 思考折算：${fits}/${cs.length} 不超时` : '') + `</span></td>`
      : `<td class="cell"><span class="n">—</span></td>`;
  }
  total += '</tr>';
  document.getElementById('overview').innerHTML =
    `<div class="overview-scroll"><table class="overview"><thead>${head}</thead><tbody>${rows}</tbody><tfoot>${total}</tfoot></table></div>`;
}
function thumbs(list, cls) {
  return (list || []).map(t => {
    const mark = (t.u != null && t.w) ? `<i class="mark" style="left:${100 * t.u / t.w}%;top:${100 * t.v / t.h}%"></i>` : '';
    const boxes = (t.boxes && t.w) ? t.boxes.map(b => `<i class="box" style="left:${100 * b[0] / t.w}%;top:${100 * b[1] / t.h}%;width:${100 * (b[2] - b[0]) / t.w}%;height:${100 * (b[3] - b[1]) / t.h}%"><b>#${esc(b[4])}</b></i>`).join('') : '';
    return `<a class="thumb ${cls || ''}" href="${esc(t.src)}" target="_blank" rel="noopener"><img loading="lazy" src="${esc(t.src)}" alt="${esc(t.camera || '')}">${boxes}${mark}<span>${esc(t.camera || '')}</span></a>`;
  }).join('');
}
function card(c) {
  const times = c.n > 1 ? `<span class="times">×${c.n}</span>` : '';
  const code = c.kind === 'code' && c.code ? `<pre>${esc(c.code)}</pre>` : '';
  const pics = (c.thumbs && c.thumbs.length) || (c.saw && c.saw.length)
    ? `<div class="thumbs">${thumbs(c.thumbs, c.kind === 'look' ? 'small' : '')}${thumbs(c.saw, 'small')}</div>` : '';
  const res = c.result ? `<div class="res ${c.ok ? '' : 'bad'}">${esc(c.result)}</div>` : '';
  const how = c.note ? `<div class="how">${D.mode === 'agent' ? '↳ 意图：' : '↳ 坐标'}${esc(c.note)}</div>` : '';
  const inner = c.inner ? `<div class="inner">程序里调用：${esc(c.inner)}</div>` : '';
  const found = c.found ? `<div class="found">得到：${c.found.map(esc).join('；')}</div>` : '';
  const kind = D.tags[c.kind] || '·';
  const cls = ['look', 'detect', 'pose', 'lift', 'state'].includes(c.kind) ? 'sense'
            : (['step', 'move', 'point', 'grip'].includes(c.kind) ? 'act' : '');
  return `<li data-t="${c.t0}"><span class="t">${fmt(c.t0)} s</span><div class="what"><span class="tag ${cls}">${kind}</span><span class="text">${esc(c.text)}</span>${times}${how}${inner}${found}${res}${code}${pics}</div></li>`;
}
const SENSE = ['camera_look', 'vision_detect', 'vision_pose', 'vision_lift', 'robot_state',
               'camera.look', 'vision.detect', 'vision.pose', 'vision.lift', 'robot.state'];
function tally(c) {
  const chip = ([k, n]) => `<span class="chip ${SENSE.includes(k) ? 'sense' : ''}">${esc(k)} ×${n}</span>`;
  let out = (c.tally || []).map(chip).join('');
  if ((c.inner_tally || []).length) out += `<span class="sep">程序里：</span>` + c.inner_tally.map(chip).join('');
  return out;
}
function why(c) {
  const r = String(c.reason || '');
  if (r.startsWith('budget_replies')) return '超过回复上限';
  if (r.startsWith('budget_turns')) return '超过轮数上限';
  if (r.startsWith('budget_calls')) return '超过工具调用上限';
  if (r.startsWith('timeout')) return '超过时间上限';
  if (r.includes('violation')) return '违反任务规则：' + r.split(':').slice(1).join(':').split(',').map(n => D.rules[n] || n).join('、');
  if (r.includes('execution_failure')) return '没做成';
  if (r.includes('physics_flag')) return '物理审计不通过';
  return (c.score || {}).failure || r;
}
function panel(task, p) {
  const c = (D.clips[task] || {})[p];
  const [name, like, says, how] = D.names[p];
  if (!c) return `<div class="panel"><h3>${esc(name)}<i>${esc(like)}</i></h3><div class="novideo">还没有录</div></div>`;
  const verdict = c.success ? '<span class="verdict ok">✓ 成功</span>'
                            : `<span class="verdict no">✗ ${esc(why(c))}</span>`;
  const b = c.budget || {};
  const cap = (n, m) => m ? `<b>${n}</b> / ${m}` : `<b>${n}</b>`;
  const video = c.video ? `<video src="${esc(c.video)}" controls muted playsinline preload="metadata"></video>`
                        : '<div class="novideo">这一段没有画面（mock 引擎）</div>';
  return `<div class="panel" data-speed="${c.speed}" data-lead="${c.lead}">
    <h3>${esc(name)}<i>${esc(like)}</i></h3>
    <div class="stats">${verdict}<span>${cap(c.replies, b.max_replies)} 次决策</span><span><b>${c.tool_calls}</b> 次工具调用</span>
      <span>仿真 ${cap(fmt(c.sim_seconds), b.max_sim_seconds)} s</span></div>
    <div class="tally">${tally(c)}</div>
    ${video}
    <ol class="steps">${c.decisions.map(card).join('')}</ol></div>`;
}
function showTask(id, scroll) {
  const t = D.tasks[id];
  const el = document.getElementById('task');
  if (!t) { el.innerHTML = ''; return; }
  const any = Object.values(D.clips[id] || {})[0] || {};
  const hint = any.hint ? `<p><span class="label">提示</span>${esc(any.hint)}</p>` : '';
  const films = (any.demonstrations || []).length
    ? `<p><span class="label">示范</span>开始前先放 ${any.demonstrations.length} 段示范（视频开头）：${esc(any.demonstrations.map(d => d.caption).join(' / '))}</p>` : '';
  el.innerHTML = `<section class="task"><div class="task-head"><h2>${id} ${esc(t.name)}</h2>
      <nav><a href="#top">↑ 总览</a>${t.prev ? `<a href="#${t.prev}">← ${t.prev}</a>` : ''}${t.next ? `<a href="#${t.next}">${t.next} →</a>` : ''}</nav></div>
    <div class="given"><p><span class="label">指令</span>${esc(any.instruction || '')}</p>${hint}${films}</div>
    <div class="panels">${D.profiles.map(p => panel(id, p)).join('')}</div></section>`;
  for (const pn of el.querySelectorAll('.panel')) wire(pn);
  if (scroll) el.scrollIntoView({behavior: 'smooth', block: 'start'});
}
function wire(pn) {
  const v = pn.querySelector('video'); const items = [...pn.querySelectorAll('ol.steps li')];
  if (!items.length) return;
  const speed = parseFloat(pn.dataset.speed) || 2, lead = parseFloat(pn.dataset.lead) || 0;
  let current = -1;
  const mark = sim => {
    let k = -1;
    for (let i = 0; i < items.length; i++) { if (parseFloat(items[i].dataset.t) <= sim + 1e-6) k = i; else break; }
    if (k === current) return;
    if (current >= 0) items[current].classList.remove('now');
    current = k;
    if (k >= 0) {
      items[k].classList.add('now');
      const list = items[k].parentElement;
      const top = items[k].offsetTop - list.offsetTop;
      if (top < list.scrollTop || top > list.scrollTop + list.clientHeight - 40) list.scrollTop = top - 40;
    }
  };
  if (v) v.addEventListener('timeupdate', () => mark(v.currentTime * speed - lead));
  items.forEach(li => li.addEventListener('click', () => {
    const sim = parseFloat(li.dataset.t);
    if (v) { v.currentTime = Math.max(0, (sim + lead) / speed); v.play().catch(() => {}); }
    mark(sim);
  }));
}
overview();
const route = scroll => { const id = location.hash.slice(1); if (D.tasks[id]) showTask(id, scroll); else if (id !== 'top') showTask(D.first, false); };
window.addEventListener('hashchange', () => route(true)); route(false);
