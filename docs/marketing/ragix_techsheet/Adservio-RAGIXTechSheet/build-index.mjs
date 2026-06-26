#!/usr/bin/env node
/**
 * build-index.mjs — Génère index.html avec le contenu de tous les chapitres
 * embarqué directement (pas de fetch, pas d'iframe, fonctionne en file://)
 *
 * Ce script est généré par le skill adservio-report.
 * Adapter les constantes REPORT_TITLE, REPORT_SUBTITLE et le tableau nav.
 */
import { readFileSync, writeFileSync, readdirSync } from 'node:fs';
import { join, dirname, basename } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPORT_TITLE = 'RAGIX Tech Sheet';
const REPORT_SUBTITLE = 'Sovereign Document & Code Intelligence';
const REPORT_SLUG = basename(__dirname);

const files = readdirSync(__dirname)
  .filter(f => /^\d{2}_.*\.html$/.test(f))
  .sort();

/* ── Extraire le <body> de chaque fichier ──────────────────────────────── */

const chapters = files.map(f => {
  const html = readFileSync(join(__dirname, f), 'utf-8');
  const match = html.match(/<body[^>]*>([\s\S]*)<\/body>/i);
  const id = f.replace('.html', '');
  return { id, body: match ? match[1].trim() : '' };
});

/* ── Sidebar entries ───────────────────────────────────────────────────── */
// {{NAV_ENTRIES}} — adapter ce tableau au rapport
const nav = [
  { id: 'all', num: '&bull;', label: 'All chapters' },
  { sep: true },
  { id: '00_front', num: '', label: 'Cover & contents' },
  { id: '01_overview', num: '1', label: 'RAGIX in one minute' },
  { id: '02_differentiators', num: '2', label: 'What makes RAGIX unique' },
  { id: '03_capabilities', num: '3', label: 'Capability catalogue' },
  { id: '04_sovereign', num: '4', label: 'Orchestratable by an AI' },
  { id: '05_proof', num: '5', label: 'Proof at scale' },
  { id: '06_forge', num: '6', label: 'Engineering forge' },
  { id: '07_maturity', num: '7', label: 'Maturity & openness' },
];

function countPages(body) {
  return (body.match(/<div class="page[" ]/g) || []).length;
}

const navHtml = nav.map(n => {
  if (n.sep) return '    <li><div class="nav_sep"></div></li>';
  const ch = chapters.find(c => c.id === n.id);
  const pages = ch ? countPages(ch.body) : 0;
  const badge = pages ? `<span class="pages_badge">${pages}p</span>` : '';
  const active = n.id === 'all' ? ' active' : '';
  return `    <li><a class="nav_link${active}" data-id="${n.id}"><span class="nav_num">${n.num}</span> ${n.label} ${badge}</a></li>`;
}).join('\n');

/* ── Renuméroter les footer_page séquentiellement ──────────────────────── */

let pageCounter = 0;
for (const ch of chapters) {
  ch.body = ch.body.replace(/<span class="footer_page">\d+<\/span>/g, () => {
    pageCounter++;
    return `<span class="footer_page">${pageCounter}</span>`;
  });
  const pagesInChapter = countPages(ch.body);
  const footersInChapter = (ch.body.match(/<span class="footer_page">/g) || []).length;
  if (footersInChapter < pagesInChapter) {
    pageCounter += (pagesInChapter - footersInChapter);
  }
}

/* ── Renuméroter aussi dans les fichiers source ────────────────────────── */

let srcPageCounter = 0;
for (const f of files) {
  let content = readFileSync(join(__dirname, f), 'utf-8');
  const pagesInFile = (content.match(/<div class="page[" ]/g) || []).length;
  const footersInFile = (content.match(/<span class="footer_page">\d+<\/span>/g) || []).length;
  let changed = false;

  content = content.replace(/<span class="footer_page">\d+<\/span>/g, () => {
    srcPageCounter++;
    changed = true;
    return `<span class="footer_page">${srcPageCounter}</span>`;
  });

  if (footersInFile < pagesInFile) {
    srcPageCounter += (pagesInFile - footersInFile);
  }

  if (changed) writeFileSync(join(__dirname, f), content, 'utf-8');
}

console.log('Pages numerotees : ' + srcPageCounter);

/* ── Chapter blocks ────────────────────────────────────────────────────── */

const blocksHtml = chapters.map(ch =>
  `<div class="chapter_block" data-id="${ch.id}">\n${ch.body}\n</div>`
).join('\n\n');

/* ── CSS ───────────────────────────────────────────────────────────────── */

const reportCss = readFileSync(join(__dirname, 'style.css'), 'utf-8');

/* ── Assemblage ────────────────────────────────────────────────────────── */

const html = `<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${REPORT_TITLE}</title>
  <link href="https://fonts.googleapis.com/css2?family=Funnel+Display:wght@300;400;500;600;700;800&family=Funnel+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://cdn.hugeicons.com/font/hgi-stroke-rounded.css">
  <style>
${reportCss}

/* ── Override pour index ────────────────────────────────────────────── */

/* ── Index layout ──────────────────────────────────────────────────── */

html { scroll-behavior: smooth; }

body {
  background: #f1f5f9 !important;
  padding: 0 !important;
  display: flex;
  min-height: 100vh;
}

.sidebar {
  position: fixed;
  top: 0;
  left: 0;
  width: 280px;
  height: 100vh;
  background: #ffffff;
  border-right: 1px solid #e2e8f0;
  overflow-y: auto;
  padding: 32px 0;
  z-index: 100;
  flex-shrink: 0;
}

.sidebar_header { padding: 0 24px 24px; }

.sidebar_logo {
  height: 24px;
  width: auto;
  margin-bottom: 16px;
  display: block;
}

.sidebar_title {
  font-family: 'Funnel Display', sans-serif;
  font-size: 16px;
  font-weight: 600;
  color: #000;
  line-height: 1.2;
}

.sidebar_subtitle {
  font-size: 11px;
  color: #9ca3af;
  margin-top: 4px;
}

.nav_list { list-style: none; }

.nav_link {
  display: flex;
  align-items: baseline;
  gap: 10px;
  padding: 8px 24px;
  font-size: 12px;
  color: #64748b;
  text-decoration: none;
  cursor: pointer;
  transition: background 0.15s, color 0.15s;
}

.nav_link:hover { background: #f0f7ff; color: #1e40af; }
.nav_link.active { background: #eff6ff; color: #2563eb; font-weight: 600; }

.nav_num {
  font-size: 11px;
  font-weight: 600;
  color: #2563eb;
  min-width: 18px;
  text-align: right;
}

.nav_sep { height: 1px; background: #e2e8f0; margin: 8px 24px; }

.pages_badge {
  font-size: 10px;
  color: #9ca3af;
  margin-left: auto;
  flex-shrink: 0;
}

.merge_bar { padding: 16px 24px; margin-top: 16px; }

.merge_btn {
  display: block;
  width: 100%;
  padding: 10px 16px;
  font-family: 'Funnel Sans', sans-serif;
  font-size: 12px;
  font-weight: 600;
  color: #fff;
  background: #2563eb;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  text-align: center;
  text-decoration: none;
}

.merge_btn:hover { background: #1d4ed8; }

.main {
  margin-left: 280px;
  flex: 1;
  padding: 40px 0;
}

.chapter_block.hidden { display: none; }

@media print {
  .sidebar { display: none; }
  .main { margin-left: 0; }
}
  </style>
</head>
<body>

<nav class="sidebar">
  <div class="sidebar_header">
    <img class="sidebar_logo" src="../img/logo-adservio-h.png" alt="Adservio">
    <div class="sidebar_title">${REPORT_TITLE}</div>
    <div class="sidebar_subtitle">${REPORT_SUBTITLE}</div>
  </div>

  <ul class="nav_list" id="nav">
\${navHtml}
  </ul>

  <div class="merge_bar">
    <a class="merge_btn" href="../${REPORT_SLUG}.html" target="_blank">Ouvrir le rapport fusionné</a>
  </div>
</nav>

<main class="main">
\${blocksHtml}
</main>

<script>
document.getElementById('nav').addEventListener('click', function (e) {
  var link = e.target.closest('.nav_link');
  if (!link) return;
  var id = link.dataset.id;

  document.querySelectorAll('.nav_link').forEach(function (el) {
    el.classList.toggle('active', el.dataset.id === id);
  });

  document.querySelectorAll('.chapter_block').forEach(function (el) {
    if (id === 'all') {
      el.classList.remove('hidden');
    } else {
      el.classList.toggle('hidden', el.dataset.id !== id);
    }
  });

  window.scrollTo({ top: 0 });
});
</script>

</body>
</html>`;

writeFileSync(join(__dirname, 'index.html'), html, 'utf-8');
console.log('index.html genere (' + chapters.length + ' chapitres embarques)');
