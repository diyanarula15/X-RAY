import { chromium } from 'playwright';
const b = await chromium.launch({ args: ['--use-gl=angle','--enable-unsafe-swiftshader','--ignore-gpu-blocklist'] });
const p = await b.newPage({ viewport: { width: 1600, height: 950 } });
await p.goto('http://127.0.0.1:8011/', { waitUntil: 'networkidle', timeout: 90000 });
await p.evaluate(() => localStorage.setItem('xray-seen','1'));
await p.reload({ waitUntil: 'networkidle' });
await p.waitForTimeout(7000);
// press play, then measure frame intervals for 6 s
await p.getByRole('button', { name: '▶' }).click().catch(()=>{});
await p.waitForTimeout(1200);
const stats = await p.evaluate(() => new Promise((res) => {
  const ts = []; let last = performance.now(); let n = 0;
  const tick = () => {
    const now = performance.now(); ts.push(now - last); last = now; n++;
    if (n < 360) requestAnimationFrame(tick);
    else {
      ts.sort((a,b)=>a-b);
      const q = (f) => ts[Math.floor(f*(ts.length-1))];
      res({ n: ts.length, median: +q(0.5).toFixed(2), p90: +q(0.9).toFixed(2),
            p99: +q(0.99).toFixed(2), max: +ts[ts.length-1].toFixed(2),
            over33: ts.filter(x=>x>33).length });
    }
  };
  requestAnimationFrame(tick);
}));
console.log('frame intervals (ms):', JSON.stringify(stats));
console.log('=> median fps', (1000/stats.median).toFixed(1), '| dropped frames >33ms:', stats.over33, '/', stats.n);
await p.screenshot({ path: '/Users/apple/Desktop/x-ray/out/shots/p1-play.png' });
await b.close();
