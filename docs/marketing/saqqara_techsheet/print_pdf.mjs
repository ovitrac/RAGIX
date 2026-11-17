// print_pdf.mjs <in.html> <out.pdf>: print an A4 tech sheet with headless Chrome.
// Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
import { createRequire } from 'node:module';
import { resolve } from 'node:path';
import { homedir } from 'node:os';
const require = createRequire(resolve(homedir(), 'node_modules') + '/');
const puppeteer = require('puppeteer-core');
const [inp, out] = process.argv.slice(2);
const browser = await puppeteer.launch({ executablePath: '/usr/bin/google-chrome', headless: true, args: ['--no-sandbox', '--font-render-hinting=none'] });
const page = await browser.newPage();
await page.goto('file://' + resolve(inp), { waitUntil: 'networkidle0' });
await page.evaluate(() => document.fonts.ready);
await page.pdf({ path: out, format: 'A4', printBackground: true, preferCSSPageSize: true, margin: { top: 0, right: 0, bottom: 0, left: 0 } });
await browser.close();
