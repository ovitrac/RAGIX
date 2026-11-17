#!/usr/bin/env node
/**
 * print_slides.mjs — Print the deck to one PDF, one 1280 x 720 pt page per slide, text kept as text.
 *
 * Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
 *
 * Each slide file is opened in headless Chrome at 1280 x 720 px and printed at scale 4/3, which fills a
 * 1280 x 720 pt page (the page size of the RAGIX slides PDF). The pages are joined with pdfunite in the
 * order of deploy.py's SLIDE_ORDER.
 *
 * Usage: node print_slides.mjs <slides_dir> <out.pdf>
 * Needs puppeteer-core (resolved from the user's node_modules), /usr/bin/google-chrome and pdfunite.
 */
import { createRequire } from 'node:module';
import { resolve, join, dirname } from 'node:path';
import { readFileSync, mkdtempSync, rmSync } from 'node:fs';
import { homedir, tmpdir } from 'node:os';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const require = createRequire(resolve(homedir(), 'node_modules') + '/');
const puppeteer = require('puppeteer-core');

const [dir, out] = process.argv.slice(2);
if (!dir || !out) { console.error('usage: node print_slides.mjs <slides_dir> <out.pdf>'); process.exit(2); }

const here = dirname(fileURLToPath(import.meta.url));
const deploy = readFileSync(join(here, 'deploy.py'), 'utf-8');
const block = deploy.match(/SLIDE_ORDER = \[([\s\S]*?)\]/);
if (!block) { console.error('SLIDE_ORDER not found in deploy.py'); process.exit(2); }
const order = [...block[1].matchAll(/"([^"]+\.html)"/g)].map(m => m[1]);

const work = mkdtempSync(join(tmpdir(), 'slides-'));
const browser = await puppeteer.launch({ executablePath: '/usr/bin/google-chrome', headless: true,
  args: ['--no-sandbox', '--font-render-hinting=none'] });
const page = await browser.newPage();
await page.setViewport({ width: 1280, height: 720, deviceScaleFactor: 1 });
const parts = [];
for (const [i, f] of order.entries()) {
  await page.goto('file://' + resolve(join(dir, f)), { waitUntil: 'networkidle0' });
  await page.evaluate(() => document.fonts.ready);
  const part = join(work, `${String(i + 1).padStart(2, '0')}.pdf`);
  await page.pdf({ path: part, width: `${1280 / 72}in`, height: `${720 / 72}in`, scale: 4 / 3, printBackground: true,
    margin: { top: 0, right: 0, bottom: 0, left: 0 }, pageRanges: '1' });
  parts.push(part);
}
await browser.close();
execFileSync('pdfunite', [...parts, resolve(out)]);
rmSync(work, { recursive: true, force: true });
console.log(`${parts.length} slides -> ${out}`);
