import { chromium } from 'playwright';
const OUT = '/Users/apple/Desktop/x-ray/out/shots';
const b = await chromium.launch({ args: ['--use-gl=angle','--enable-unsafe-swiftshader','--ignore-gpu-blocklist'] });
const p = await b.newPage({ viewport: { width: 1600, height: 950 } });
const errs = [];
p.on('pageerror', e => errs.push('PAGEERROR: ' + e.message));
p.on('console', m => { if (m.type()==='error') errs.push('CONSOLE: ' + m.text().slice(0,200)); });
await p.goto('http://127.0.0.1:8011/', { waitUntil: 'networkidle', timeout: 90000 });
await p.waitForTimeout(3000);
await p.screenshot({ path: `${OUT}/e0-onboard.png` });          // first-run overlay
for (let i=0;i<4;i++){ try{ await p.getByRole('button',{name:/next|start/}).click(); await p.waitForTimeout(500);}catch(e){} }
await p.waitForTimeout(6500);
await p.screenshot({ path: `${OUT}/e1-chase.png` });
try { await p.getByRole('button',{name:'duel',exact:true}).click(); await p.waitForTimeout(3500);
      await p.screenshot({ path: `${OUT}/e2-duel.png` }); } catch(e){ errs.push('duel'); }
try { await p.getByRole('button',{name:'?',exact:true}).click(); await p.waitForTimeout(1200);
      await p.screenshot({ path: `${OUT}/e3-help.png` });
      await p.getByRole('button',{name:'got it'}).click(); } catch(e){ errs.push('help '+e.message.slice(0,60)); }
console.log(errs.length ? errs.join('\n') : 'no console errors');
await b.close();
