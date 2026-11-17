#!/usr/bin/env node
/**
 * check_pages.mjs — Measure every A4 page of a tech-sheet HTML file and report overflow.
 *
 * Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
 *
 * The .page boxes are fixed at 210 x 297 mm with overflow hidden, so content that runs past the
 * footer is clipped without any warning. This script opens the file in headless Chrome, waits for
 * the fonts, and for each page compares the lowest content element with the top of the footer.
 * It prints one line per page and exits 1 if any page overflows. With an output directory, it also
 * saves one PNG per page.
 *
 * Usage: node check_pages.mjs <file.html> [png_dir]
 * Needs puppeteer-core (resolved from the user's node_modules) and /usr/bin/google-chrome.
 */
import { createRequire } from 'node:module';
import { resolve } from 'node:path';
import { mkdirSync } from 'node:fs';
import { homedir } from 'node:os';

const require = createRequire(resolve(homedir(), 'node_modules') + '/');
const puppeteer = require('puppeteer-core');

const [file, pngDir] = process.argv.slice(2);
if (!file) { console.error('usage: node check_pages.mjs <file.html> [png_dir]'); process.exit(2); }

const browser = await puppeteer.launch({ executablePath: '/usr/bin/google-chrome', headless: true,
  args: ['--no-sandbox', '--font-render-hinting=none'] });
const page = await browser.newPage();
await page.setViewport({ width: 1000, height: 1400, deviceScaleFactor: 1 });
await page.goto('file://' + resolve(file), { waitUntil: 'networkidle0' });
await page.evaluate(async () => { await document.fonts.ready; });

const rows = await page.evaluate(() => {
  const mm = px => (px * 25.4 / 96).toFixed(1);
  return [...document.querySelectorAll('.page')].map((p, i) => {
    const pr = p.getBoundingClientRect();
    const footer = p.querySelector(':scope > .page_footer');
    const limit = footer ? footer.getBoundingClientRect().top : pr.bottom - 60;
    let low = pr.top, lowEl = '';
    p.querySelectorAll(':scope > *').forEach(el => {
      if (el.classList.contains('page_footer')) return;
      const b = el.getBoundingClientRect().bottom;
      if (b > low) { low = b; lowEl = el.className || el.tagName.toLowerCase(); }
    });
    return { n: i + 1, used: mm(low - pr.top), limit: mm(limit - pr.top), slack: mm(limit - low),
             over: low > limit - 4, lowEl };
  });
});

let bad = 0;
for (const r of rows) {
  if (r.over) bad++;
  console.log(`page ${String(r.n).padStart(2)}  content to ${r.used} mm  footer at ${r.limit} mm  ` +
              `slack ${r.slack} mm  ${r.over ? 'OVERFLOW' : 'ok'}  (lowest: ${r.lowEl})`);
}
if (pngDir) {
  mkdirSync(pngDir, { recursive: true });
  const els = await page.$$('.page');
  for (let i = 0; i < els.length; i++) {
    await els[i].screenshot({ path: `${pngDir}/page_${String(i + 1).padStart(2, '0')}.png` });
  }
}
await browser.close();
process.exit(bad ? 1 : 0);
