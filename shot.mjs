import { chromium } from 'playwright';
const OUT = '/Users/apple/Desktop/x-ray/out/shots';
const b = await chromium.launch({ args: ['--use-gl=angle','--enable-unsafe-swiftshader','--ignore-gpu-blocklist'] });
const p = await b.newPage({ viewport: { width: 1600, height: 950 } });
const errs = [];
p.on('pageerror', e => errs.push('PAGEERROR: ' + e.message));
p.on('console', m => { if (m.type()==='error') errs.push('CONSOLE: ' + m.text().slice(0,200)); });
await p.goto('http://127.0.0.1:8011/', { waitUntil: 'networkidle', timeout: 90000 });
await p.waitForTimeout(9000);
await p.screenshot({ path: `${OUT}/r1-chase.png` });
for (const [btn, name, wait] of [['duel','r2-duel',3500],['deployment','r3-deploy',2500],
                                 ['tactical','r4-tactical',3500]]) {
  try { await p.getByRole('button',{name:btn,exact:true}).click(); await p.waitForTimeout(wait);
        await p.screenshot({ path: `${OUT}/${name}.png` }); }
  catch(e){ errs.push(`${btn}: ${e.message.slice(0,90)}`); }
}
console.log(errs.length ? errs.join('\n') : 'no console errors');
await b.close();
