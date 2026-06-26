#!/usr/bin/env node
/**
 * merge.mjs — Fusionne les fichiers chapitre en un rapport HTML unique
 *
 * Usage : node merge.mjs
 *
 * Ce script est généré par le skill adservio-report.
 * Adapter la constante REPORT_TITLE et le tableau tocEntries au rapport.
 */
import { readFileSync, writeFileSync, readdirSync } from 'node:fs';
import { join, dirname, basename } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPORT_TITLE = 'RAGIX — Sovereign Document & Code Intelligence';
const REPORT_SLUG = basename(__dirname);

/* ── 1. Lister les fichiers chapitre ───────────────────────────────────── */

const files = readdirSync(__dirname)
  .filter(f => /^\d{2}_.*\.html$/.test(f))
  .sort();

console.log(`Fichiers a fusionner : ${files.length}`);
files.forEach(f => console.log(`  ${f}`));

/* ── 2. Extraire les pages de chaque fichier ───────────────────────────── */

function extractPages(html) {
  const lines = html.split('\n');
  const pages = [];
  let buf = null;
  let depth = 0;

  for (const line of lines) {
    if (/^<div class="page[" ]/.test(line)) {
      buf = [];
      depth = 0;
    }
    if (buf !== null) {
      buf.push(line);
      const opens = (line.match(/<div[\s>]/g) || []).length;
      const closes = (line.match(/<\/div>/g) || []).length;
      depth += opens - closes;
      if (depth === 0) {
        pages.push(buf.join('\n'));
        buf = null;
      }
    }
  }
  return pages;
}

const allPages = [];
for (const f of files) {
  const html = readFileSync(join(__dirname, f), 'utf-8');
  const pages = extractPages(html);
  console.log(`  ${f} : ${pages.length} page(s)`);
  allPages.push(...pages);
}

console.log(`\nTotal pages : ${allPages.length}`);

/* ── 3. Corriger les chemins d'images ──────────────────────────────────── */

for (let i = 0; i < allPages.length; i++) {
  allPages[i] = allPages[i]
    .replace(/src="\.\.\/diagrams\//g, 'src="diagrams/')
    .replace(/src="\.\.\/img\//g, 'src="img/');
}

/* ── 4. Renuméroter les footer_page ────────────────────────────────────── */

for (let i = 0; i < allPages.length; i++) {
  const pageNum = i + 1;
  allPages[i] = allPages[i].replace(
    /<span class="footer_page">\d+<\/span>/,
    `<span class="footer_page">${pageNum}</span>`
  );
}

/* ── 5. Mettre à jour la TOC ───────────────────────────────────────────── */

// {{TOC_ENTRIES}} — adapter ce tableau au rapport
const tocEntries = [
  { label: '1. RAGIX in one minute &mdash; the problem we solve', marker: 'SECTION 1</span>' },
  { label: '2. What makes RAGIX unique &mdash; five differentiators', marker: 'SECTION 2</span>' },
  { label: '3. The capability catalogue &mdash; six kernel families', marker: 'SECTION 3</span>' },
  { label: '4. Orchestratable by an AI &mdash; sovereignly, with proof', marker: 'SECTION 4</span>' },
  { label: '5. Proof at scale &mdash; field case studies', marker: 'SECTION 5</span>' },
  { label: '6. Operating an advanced engineering forge', marker: 'SECTION 6</span>' },
  { label: '7. Maturity, openness &amp; technology stack', marker: 'SECTION 7</span>' },
];

const sectionPages = {};
for (const entry of tocEntries) {
  for (let i = 0; i < allPages.length; i++) {
    if (allPages[i].includes(entry.marker)) {
      sectionPages[entry.label] = i + 1;
      break;
    }
  }
}

// Mettre à jour les numéros dans la page TOC (page index 1)
if (allPages.length > 1) {
  let tocPage = allPages[1];
  for (const entry of tocEntries) {
    const pageNum = sectionPages[entry.label];
    if (!pageNum) continue;
    const escaped = entry.label.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const re = new RegExp(
      `(<span class="toc_label">${escaped}</span>[\\s\\S]*?<span class="toc_page">)\\d+(</span>)`
    );
    tocPage = tocPage.replace(re, `$1${pageNum}$2`);
  }
  allPages[1] = tocPage;
}

/* ── 6. Assembler le HTML final ────────────────────────────────────────── */

const css = readFileSync(join(__dirname, 'style.css'), 'utf-8');

const output = `<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>${REPORT_TITLE}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Funnel+Display:wght@300;400;500;600;700;800&family=Funnel+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdn.hugeicons.com/font/hgi-stroke-rounded.css">
<style>
${css}
</style>
</head>
<body>

${allPages.join('\n\n')}

</body>
</html>`;

const outPath = join(__dirname, '..', REPORT_SLUG + '.html');
writeFileSync(outPath, output, 'utf-8');
console.log(`\nRapport fusionne ecrit : ${outPath}`);

/* ── 7. Afficher le mapping TOC ────────────────────────────────────────── */

if (tocEntries.length > 0) {
  console.log('\nTable des matieres :');
  for (const entry of tocEntries) {
    const p = sectionPages[entry.label] || '?';
    console.log(`  p.${String(p).padStart(2)} — ${entry.label}`);
  }
}
