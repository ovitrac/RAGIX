#!/usr/bin/env python3
"""
deploy.py — Build fully self-contained, offline RAGIX tech-sheet deliverables.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

Produces, under this directory:
  - RAGIX-Tech-Sheet-Document.html      single-file A4 dossier  (CSS + images + fonts + icons embedded)
  - RAGIX-Tech-Sheet-Presentation.html  single-file 14-slide deck (each slide isolated in an iframe; fonts/icons embedded once)
  - RAGIX-Tech-Sheet-bundle.zip         the two HTML files + a short README

Fonts/icons are fetched once from their CDNs and cached in .fontcache/ so the
build is repeatable offline afterwards. The generated HTML needs NO network.

Usage:  python3 deploy.py
"""
import base64, glob, json, mimetypes, os, re, sys, urllib.request, zipfile
from urllib.parse import unquote
from io import BytesIO

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.join(HERE, "ragix-tech-sheet")            # slides project
DOSSIER = os.path.join(HERE, "Adservio-RAGIXTechSheet.html")
CACHE = os.path.join(HERE, ".fontcache")
os.makedirs(CACHE, exist_ok=True)

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"}

SLIDE_ORDER = [
    "slide-01-couverture.html", "slide-01b-sommaire.html", "slide-02-defi.html",
    "slide-03-plateforme.html", "slide-04-differenciateurs.html",
    "slide-05-faits-documents.html", "slide-06-generation-pipeline.html",
    "slide-07-capacites.html", "slide-08-raisonnement.html", "slide-09-souverain.html",
    "slide-10-preuve.html", "slide-11-cas.html", "slide-12-forge.html",
    "slide-13-cloture.html",
]

# ----------------------------------------------------------------------------- fetch + cache
def _cache_get(url):
    key = re.sub(r"[^A-Za-z0-9._-]", "_", url)[-180:]
    path = os.path.join(CACHE, key)
    if os.path.exists(path):
        return open(path, "rb").read()
    data = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40).read()
    open(path, "wb").write(data)
    return data

def _data_uri(raw, mime):
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"

# ----------------------------------------------------------------------------- inline a webfont CSS (Google / Hugeicons)
def keep_latin(css):
    """Keep only the `latin` @font-face blocks (covers ASCII + accents up to U+00FF)."""
    tokens = re.findall(r'/\*\s*([a-z0-9-]+)\s*\*/\s*(@font-face\s*\{.*?\})', css, re.S)
    if not tokens:
        return css
    return "".join(f"/* {n} */\n{b}\n" for n, b in tokens if n == "latin")

def _subset_woff2(raw, unicodes):
    """Subset a woff2 font to the given codepoints; return woff2 bytes (or original on failure)."""
    try:
        from fontTools import subset
        from fontTools.ttLib import TTFont
        font = TTFont(BytesIO(raw))
        ss = subset.Subsetter(subset.Options(flavor="woff2", desubroutinize=True))
        ss.populate(unicodes=list(unicodes))
        ss.subset(font)
        buf = BytesIO(); font.flavor = "woff2"; font.save(buf)
        return buf.getvalue()
    except Exception as e:
        print(f"   ! font subset skipped ({e})")
        return raw

def inline_font_css(css_url, latin_only=False, subset_unicodes=None):
    """Fetch a font CSS, keep only woff2 sources (optionally subset them), and inline as base64."""
    css = _cache_get(css_url).decode("utf-8", "replace")
    if latin_only:
        css = keep_latin(css)
    # Drop non-woff2 font sources (.eot/.ttf/.svg/.woff) — keeps files small; woff2 is universal.
    css = re.sub(r'url\([^)]*\.(?:eot|ttf|svg|woff(?!2))[^)]*\)\s*format\([^)]*\)\s*,?', '', css, flags=re.I)
    css = re.sub(r'url\([^)]*\.eot[^)]*\)\s*,?', '', css, flags=re.I)   # bare IE eot
    css = re.sub(r',\s*;', ';', css)                                    # tidy trailing commas
    import urllib.parse  # local import for urljoin
    def repl(m):
        ref = m.group(1).strip("'\"")
        if ref.startswith("data:"):
            return m.group(0)
        full = ref if ref.startswith("http") else urllib.parse.urljoin(css_url, ref)
        raw = _cache_get(full)
        if subset_unicodes and full.endswith(".woff2"):
            raw = _subset_woff2(raw, subset_unicodes)
        mime = "font/woff2" if full.endswith(".woff2") else "application/octet-stream"
        return f"url({_data_uri(raw, mime)})"
    return re.sub(r"url\(([^)]+)\)", repl, css)

def text_unicodes(*htmls):
    """Collect the set of codepoints used in the visible text of given HTML strings (+ printable ASCII)."""
    import html as _h
    chars = set(range(0x20, 0x7F))                       # base printable ASCII
    for h in htmls:
        txt = re.sub(r"<[^>]+>", " ", h)                 # strip tags
        txt = _h.unescape(txt)
        chars.update(ord(c) for c in txt)
    chars.update(ord(c) for c in "—–‘’“”…•·→←&é è ê à â ç î ï ô û ù")  # safety: dashes, quotes, accents
    return chars

# ----------------------------------------------------------------------------- subset the Hugeicons font to only the glyphs actually used
HUGEICONS_CSS_URL = "https://cdn.hugeicons.com/font/hgi-stroke-rounded.css"

def used_hgi_classes(html):
    return set(re.findall(r'hgi-[a-z0-9-]+', html)) - {"hgi-stroke", "hgi-stroke-rounded"}

def hugeicons_min_css(html):
    """Return a minimal Hugeicons CSS: @font-face with a subset woff2 (data URI) + only the used glyph rules."""
    from fontTools import subset
    from fontTools.ttLib import TTFont
    full = _cache_get(HUGEICONS_CSS_URL).decode("utf-8", "replace")
    classes = used_hgi_classes(html)
    rules, chars = [], set()
    for cls in sorted(classes):
        m = re.search(r'\.hgi-stroke\.' + re.escape(cls) + r':before\{content:"(.+?)"\}', full)
        if not m:
            continue
        ch = m.group(1)
        chars.add(ch)
        rules.append(f'.hgi-stroke.{cls}:before{{content:"{ch}"}}')
    # locate the cached woff2 and subset it
    woff2 = next(p for p in glob.glob(os.path.join(CACHE, "*hgi-stroke-rounded.woff2*")))
    font = TTFont(woff2)
    ss = subset.Subsetter(subset.Options(flavor="woff2", desubroutinize=True, layout_features=[]))
    ss.populate(unicodes=[ord(c) for c in chars])
    ss.subset(font)
    buf = BytesIO(); font.flavor = "woff2"; font.save(buf)
    uri = _data_uri(buf.getvalue(), "font/woff2")
    base = ('@font-face{font-family:"hgi-stroke-rounded";font-display:block;'
            f'src:url({uri}) format("woff2")}}'
            '.hgi-stroke{font-family:"hgi-stroke-rounded"!important;font-style:normal;'
            '-webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale;position:relative}')
    return base + "".join(rules)

# ----------------------------------------------------------------------------- embed local images referenced by src="..."
def embed_images(html, base_dir):
    def repl(m):
        attr, q, ref = m.group(1), m.group(2), m.group(3)
        if ref.startswith(("data:", "http://", "https://", "#")):
            return m.group(0)
        rel = unquote(ref)
        path = os.path.normpath(os.path.join(base_dir, rel))
        if not os.path.exists(path):
            print(f"   ! image not found, left as-is: {ref}")
            return m.group(0)
        mime = mimetypes.guess_type(path)[0] or "image/png"
        return f'{attr}={q}{_data_uri(open(path, "rb").read(), mime)}{q}'
    return re.sub(r'(src|href)=(["\'])([^"\']+\.(?:png|jpe?g|webp|gif|svg))\2', repl, html, flags=re.I)

# ----------------------------------------------------------------------------- replace a <link rel=stylesheet href=URL> with inline <style>
def inline_link(html, url_substring, css_text):
    pat = re.compile(r'<link\b[^>]*href="[^"]*' + re.escape(url_substring) + r'[^"]*"[^>]*>', re.I)
    return pat.sub(f"<style>\n{css_text}\n</style>", html, count=1)

def extract_google_href(html, family_token):
    m = re.search(r'href="(https://fonts\.googleapis\.com/css2\?[^"]*' + re.escape(family_token) + r'[^"]*)"', html)
    return m.group(1).replace("&amp;", "&") if m else None

# ----------------------------------------------------------------------------- build the dossier (single file)
def build_dossier():
    print(" • Document (A4 dossier)")
    html = open(DOSSIER, encoding="utf-8").read()
    html = embed_images(html, HERE)                                   # img/hero.png, img/logo-adservio-h.png
    g = extract_google_href(html, "Funnel")
    if g:
        used = text_unicodes(html)
        html = inline_link(html, "fonts.googleapis.com/css2",
                           inline_font_css(g, latin_only=True, subset_unicodes=used))
    html = re.sub(r'<link\b[^>]*href="https://fonts\.googleapis\.com"[^>]*>', "", html)  # drop preconnect
    html = re.sub(r'<link\b[^>]*href="https://fonts\.gstatic\.com"[^>]*>', "", html)
    html = inline_link(html, "cdn.hugeicons.com", hugeicons_min_css(html))
    out = os.path.join(HERE, "RAGIX-Tech-Sheet-Document.html")
    open(out, "w", encoding="utf-8").write(html)
    print(f"   -> {os.path.basename(out)}  ({len(html)//1024} KB)")
    return out

# ----------------------------------------------------------------------------- build one fully self-contained slide doc (fonts embedded in its own head)
def prepare_slide(fname, google_css):
    sdir = os.path.join(PROJ, "slides")
    html = open(os.path.join(sdir, fname), encoding="utf-8").read()
    # inline slide-common.css
    css = open(os.path.join(PROJ, "library", "slide-common.css"), encoding="utf-8").read()
    html = re.sub(r'<link\b[^>]*href="\.\./library/slide-common\.css"[^>]*>',
                  f"<style>\n{css}\n</style>", html, count=1)
    # remove font/icon CDN links + preconnects (we embed our own below)
    html = re.sub(r'<link\b[^>]*(fonts\.googleapis\.com|fonts\.gstatic\.com|cdn\.hugeicons\.com)[^>]*>', "", html)
    # remove slide navigation script (parent viewer handles navigation)
    html = re.sub(r'<script\b[^>]*slide-common\.js[^>]*>\s*</script>', "", html)
    # embed images (cover photo + logo) relative to slides/ dir
    html = embed_images(html, sdir)
    # embed fonts (Instrument Serif + Work Sans) and the subset icon font directly in the head
    fonts = "<style>" + google_css + "\n" + hugeicons_min_css(html) + "</style>"
    html = html.replace("</head>", fonts + "\n</head>", 1)
    return html

# ----------------------------------------------------------------------------- build the presentation (single file, iframe-isolated self-contained slides)
def build_presentation():
    print(" • Presentation (14-slide deck)")
    raws = [open(os.path.join(PROJ, "slides", f), encoding="utf-8").read() for f in SLIDE_ORDER]
    g = extract_google_href(raws[0], "Instrument")
    used = text_unicodes(*raws)
    google_css = inline_font_css(g, latin_only=True, subset_unicodes=used) if g else ""
    slides = [prepare_slide(f, google_css) for f in SLIDE_ORDER]
    html = (VIEWER_TEMPLATE
            .replace("__SLIDES__", json.dumps(slides))
            .replace("__COUNT__", str(len(slides))))
    out = os.path.join(HERE, "RAGIX-Tech-Sheet-Presentation.html")
    open(out, "w", encoding="utf-8").write(html)
    print(f"   -> {os.path.basename(out)}  ({len(html)//1024} KB, {len(slides)} slides)")
    return out

VIEWER_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RAGIX — Tech Sheet (Presentation)</title>
<style>
  html,body{margin:0;height:100%;background:#0b1220;overflow:hidden;font-family:system-ui,sans-serif;}
  #stage{position:fixed;inset:0;display:flex;align-items:center;justify-content:center;}
  #scaler{width:1280px;height:720px;transform-origin:center center;box-shadow:0 10px 40px rgba(0,0,0,.45);background:#fff;}
  #frame{width:1280px;height:720px;border:0;display:block;background:#fff;}
  #bar{position:fixed;left:0;right:0;bottom:0;height:38px;display:flex;align-items:center;gap:14px;
       padding:0 16px;background:rgba(8,14,28,.82);color:#cbd5e1;font-size:13px;backdrop-filter:blur(6px);}
  #bar b{color:#fff;font-weight:600;}
  #bar .sp{flex:1;}
  #bar button{background:#1e293b;color:#e2e8f0;border:1px solid #334155;border-radius:6px;
              padding:5px 11px;font-size:13px;cursor:pointer;}
  #bar button:hover{background:#334155;}
  #hint{color:#64748b;}
  @media print{#bar{display:none;}}
</style>
</head>
<body>
  <div id="stage"><div id="scaler"><iframe id="frame" title="RAGIX slide"></iframe></div></div>
  <div id="bar">
    <b>RAGIX — Tech Sheet</b>
    <button id="prev">&larr; Prev</button>
    <span><span id="cur">1</span> / __COUNT__</span>
    <button id="next">Next &rarr;</button>
    <span class="sp"></span>
    <span id="hint">&larr; &rarr; navigate &middot; F fullscreen</span>
    <button id="fs">Fullscreen</button>
  </div>
<script>
  var SLIDES = __SLIDES__;
  var i = 0;
  var frame = document.getElementById('frame');
  var cur = document.getElementById('cur');
  function show(n){
    i = (n + SLIDES.length) % SLIDES.length;
    cur.textContent = (i+1);
    frame.srcdoc = SLIDES[i];
  }
  function fit(){
    var s = Math.min(window.innerWidth/1280, (window.innerHeight-38)/720);
    document.getElementById('scaler').style.transform = 'scale('+s+')';
  }
  document.getElementById('prev').onclick=function(){show(i-1);};
  document.getElementById('next').onclick=function(){show(i+1);};
  document.getElementById('fs').onclick=function(){
    if(!document.fullscreenElement){document.documentElement.requestFullscreen();}else{document.exitFullscreen();}
  };
  window.addEventListener('keydown',function(e){
    if(e.key==='ArrowRight'||e.key===' '||e.key==='PageDown'){show(i+1);e.preventDefault();}
    else if(e.key==='ArrowLeft'||e.key==='PageUp'){show(i-1);e.preventDefault();}
    else if(e.key==='Home'){show(0);}
    else if(e.key==='End'){show(SLIDES.length-1);}
    else if(e.key==='f'||e.key==='F'){document.getElementById('fs').onclick();}
  });
  window.addEventListener('resize',fit);
  fit(); show(0);
</script>
</body>
</html>
"""

README = """RAGIX — Tech Sheet (Adservio Innovation Lab)
============================================

This bundle contains two fully self-contained, offline HTML files. No internet
connection is required; fonts, icons, images and styles are all embedded.

  RAGIX-Tech-Sheet-Document.html      The A4 capability dossier (open in any browser;
                                      use Ctrl+P -> Save as PDF to print).
  RAGIX-Tech-Sheet-Presentation.html  The slide deck. Open in a browser, then:
                                        Left / Right arrows  -> previous / next slide
                                        F                    -> fullscreen
                                        Home / End           -> first / last slide

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

def build_zip(files):
    out = os.path.join(HERE, "RAGIX-Tech-Sheet-bundle.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, os.path.basename(f))
        z.writestr("README.txt", README)
    print(f" • Bundle -> {os.path.basename(out)}  ({os.path.getsize(out)//1024} KB)")
    return out

def main():
    print("Building offline RAGIX tech-sheet deliverables…")
    doc = build_dossier()
    pres = build_presentation()
    build_zip([doc, pres])
    print("Done.")

if __name__ == "__main__":
    main()
