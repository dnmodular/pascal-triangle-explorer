(() => {
  const $ = id => document.getElementById(id);
  const canvas = $('canvas');
  const ctx = canvas.getContext('2d');
  const state = { playing: false, phase: 0, last: 0, raf: null };

  const controls = ['rows','modulus','offset','scale','palette','showLines','glow','speed','direction'];
  function val(id) { return $(id).type === 'checkbox' ? $(id).checked : $(id).value; }
  function syncLabels() {
    $('rowsOut').textContent = val('rows');
    $('modOut').textContent = val('modulus');
    $('offsetOut').textContent = Number(val('offset')).toFixed(2);
    $('scaleOut').textContent = val('scale');
    $('speedOut').textContent = Number(val('speed')).toFixed(1) + '×';
    $('stats').textContent = `Rows: ${val('rows')} · Modulus: ${val('modulus')} · Offset: ${Number(val('offset')).toFixed(2)}`;
  }
  function paletteColor(t, palette) {
    t = Math.max(0, Math.min(1, t));
    const stops = {
      ember: [[17,19,26],[92,39,45],[255,111,74],[255,226,155]],
      ocean: [[10,19,31],[22,76,109],[54,190,201],[197,255,239]],
      moss: [[16,24,20],[46,88,57],[131,178,92],[231,235,164]],
      mono: [[16,17,22],[80,83,94],[170,174,185],[255,255,255]]
    }[palette] || [[17,19,26],[92,39,45],[255,111,74],[255,226,155]];
    const x = t * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(x)), f = x - i;
    const c = stops[i].map((v,k) => Math.round(v*(1-f) + stops[i+1][k]*f));
    return `rgb(${c[0]},${c[1]},${c[2]})`;
  }
  function coefficientResidues(rows, modulus) {
    const triangle = [];
    let row = [1n];
    for (let r=0; r<rows; r++) {
      triangle.push(row);
      const next = [1n];
      for (let j=1; j<row.length; j++) next.push(row[j-1] + row[j]);
      next.push(1n);
      row = next;
    }
    return triangle;
  }
  function render() {
    syncLabels();
    const rows = Number(val('rows')), mod = Number(val('modulus'));
    const size = Number(val('scale')), offset = Number(val('offset'));
    const gap = Math.max(1, Math.round(size * .18));
    const cell = size;
    const width = rows * (cell + gap) + 30;
    const height = rows * (cell + gap) + 30;
    canvas.width = width; canvas.height = height;
    canvas.style.width = Math.min(width, $('canvasWrap').clientWidth - 20) + 'px';
    canvas.style.height = 'auto';
    ctx.fillStyle = '#0d0e13'; ctx.fillRect(0,0,width,height);
    const tri = coefficientResidues(rows, mod);
    const drift = state.phase;
    for (let r=0; r<rows; r++) {
      const y = 12 + r * (cell + gap);
      const rowWidth = (r+1) * (cell+gap);
      const x0 = (width-rowWidth)/2 + gap/2;
      for (let c=0; c<=r; c++) {
        const residue = Number(tri[r][c] % BigInt(mod));
        const wave = val('direction') === 'up' ? (rows-r)/rows : val('direction') === 'pulse' ? Math.sin(r*.18 + drift)*.5+.5 : r/rows;
        const frac = ((residue / mod) + offset + wave * (state.playing ? .22 : 0)) % 1;
        const t = Math.max(0.02, Math.min(1, frac < 0 ? frac + 1 : frac));
        const x = x0 + c*(cell+gap);
        ctx.fillStyle = paletteColor(t, val('palette'));
        if (val('glow') && residue === 0) {
          ctx.shadowColor = paletteColor(.8, val('palette'));
          ctx.shadowBlur = Math.min(10, cell*.8);
        } else ctx.shadowBlur = 0;
        ctx.fillRect(x,y,cell,cell);
        ctx.shadowBlur = 0;
        if (val('showLines')) {
          ctx.strokeStyle = 'rgba(235,238,250,.12)';
          ctx.lineWidth = .5;
          ctx.strokeRect(x+.25,y+.25,cell-.5,cell-.5);
        }
      }
    }
    $('status').textContent = state.playing ? 'Animating' : 'Ready';
  }
  function tick(now) {
    if (!state.playing) return;
    if (now-state.last > 40) {
      state.phase += .025 * Number(val('speed'));
      state.last = now;
      render();
    }
    state.raf = requestAnimationFrame(tick);
  }
  function togglePlay() {
    state.playing = !state.playing;
    $('playBtn').textContent = state.playing ? 'Ⅱ Pause' : '▶ Play';
    if (state.playing) { state.last = 0; state.raf = requestAnimationFrame(tick); }
    else cancelAnimationFrame(state.raf);
    render();
  }
  function exportSVG() {
    const rows = Number(val('rows')), mod = Number(val('modulus')), size = Number(val('scale')), gap = Math.max(1, Math.round(size*.18));
    const width = rows*(size+gap)+30, height = width;
    const tri = coefficientResidues(rows, mod);
    let shapes = [`<rect width="${width}" height="${height}" fill="#0d0e13"/>`];
    for (let r=0;r<rows;r++) {
      const y=12+r*(size+gap), rowWidth=(r+1)*(size+gap), x0=(width-rowWidth)/2+gap/2;
      for(let c=0;c<=r;c++){
        const residue=Number(tri[r][c]%BigInt(mod));
        const t=Math.max(.02,Math.min(1,((residue/mod)+Number(val('offset'))+1)%1));
        const color=paletteColor(t,val('palette'));
        shapes.push(`<rect x="${x0+c*(size+gap)}" y="${y}" width="${size}" height="${size}" fill="${color}"${val('showLines')?' stroke="#ffffff" stroke-opacity=".12" stroke-width=".5"':''}/>`); 
      }
    }
    const svg=`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">${shapes.join('')}</svg>`;
    const blob = new Blob([svg], {type:'image/svg+xml;charset=utf-8'});
    const a = document.createElement('a'); a.href=URL.createObjectURL(blob); a.download='pascal-square-scanner.svg'; a.click(); URL.revokeObjectURL(a.href);
  }
  controls.forEach(id => $(id).addEventListener('input', render));
  $('playBtn').addEventListener('click', togglePlay);
  $('resetBtn').addEventListener('click', () => {
    $('rows').value=64; $('modulus').value=7; $('offset').value=0; $('scale').value=8;
    $('palette').value='ember'; $('showLines').checked=true; $('glow').checked=true;
    $('speed').value=1; $('direction').value='down'; state.phase=0; render();
  });
  $('saveBtn').addEventListener('click', exportSVG);
  document.querySelectorAll('[data-mod]').forEach(btn => btn.addEventListener('click', () => {
    $('modulus').value=btn.dataset.mod; $('rows').value=btn.dataset.rows; render();
  }));
  window.addEventListener('resize', render);
  render();
})();
