// check_slides.mjs <slides_dir> <png_dir>: render each slide at 1280x720, report clipped content, save PNGs.
// Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
import { createRequire } from 'node:module';
import { resolve, join } from 'node:path';
import { readdirSync, mkdirSync } from 'node:fs';
import { homedir } from 'node:os';
const require = createRequire(resolve(homedir(), 'node_modules') + '/');
const puppeteer = require('puppeteer-core');
const [dir, out] = process.argv.slice(2);
mkdirSync(out, { recursive: true });
const browser = await puppeteer.launch({ executablePath: '/usr/bin/google-chrome', headless: true, args: ['--no-sandbox'] });
const page = await browser.newPage();
await page.setViewport({ width: 1280, height: 720, deviceScaleFactor: 1 });
let bad = 0;
for (const f of readdirSync(dir).filter(f => f.endsWith('.html')).sort()) {
  await page.goto('file://' + resolve(join(dir, f)), { waitUntil: 'networkidle0' });
  await page.evaluate(() => document.fonts.ready);
  const r = await page.evaluate(() => {
    const issues = [];
    for (const el of document.querySelectorAll('.slide-body, .card-inner, .diagram-outer, .step, .cmp, .sommaire-grid, .kpi-content')) {
      if (el.scrollHeight > el.clientHeight + 1) issues.push(`${el.className.split(' ')[0]} +${el.scrollHeight - el.clientHeight}px`);
    }
    return issues;
  });
  if (r.length) bad++;
  console.log(`${f.padEnd(34)} ${r.length ? 'CLIPPED ' + r.join(', ') : 'ok'}`);
  await page.screenshot({ path: join(out, f.replace('.html', '.png')) });
}
await browser.close();
process.exit(bad ? 1 : 0);
