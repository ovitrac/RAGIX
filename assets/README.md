# RAGIX Assets

This directory contains static assets, such as logos and images, for the RAGIX project.

**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

---

## Contents

| File | Format | Purpose |
|------|--------|---------|
| `ragix-logo.png` | PNG, 1024×1536, RGB | Master RAGIX logo, portrait, full colour |
| `ragix.png` | PNG, 1024×1024, RGB | Square RAGIX mark |
| `ragix-label-grayscale.png` | PNG, 984×1419, 8-bit grey | B/W label artwork, continuous tone |
| `ragix-label-lineart.png` | PNG, 1968×2838, 1-bit | B/W label artwork, outlines only |
| `ragix-label-halftone.png` | PNG, 1968×2838, 1-bit | B/W label artwork, dithered tone |
| `ragix.mp4`, `ragix-demo-loop.mp4` | MP4 | Demonstration clips |
| `intelligentix.png` | PNG | Companion project mark |
| `adservio-logo.svg`, `adservio-logo.png` | SVG, PNG | Adservio corporate logo |
| `adservio-experience.jpg` | JPEG | Adservio corporate imagery |

---

## Black & white label artwork

Three monochrome renderings of `ragix-logo.png`, prepared for printing on a
portrait label. All three are trimmed to the artwork bounding box and re-padded
with a 4.5 % quiet zone, and all three carry DPI metadata giving the same
physical size:

**83.3 × 120.1 mm** (aspect 1 : 1.44 — fits a 70 × 100 mm or A6 label with
negligible rescaling).

| Variant | Mode | Pixels | DPI | Ink coverage | Intended device |
|---------|------|--------|-----|--------------|-----------------|
| `ragix-label-grayscale.png` | 8-bit grey | 984 × 1419 | 300 | 24.0 % | Laser / inkjet |
| `ragix-label-lineart.png` | 1-bit | 1968 × 2838 | 600 | 17.3 % | Thermal, engraving, stencil |
| `ragix-label-halftone.png` | 1-bit | 1968 × 2838 | 600 | 23.9 % | 1-bit devices needing tone |

### Why not a plain desaturation

The source palette contains two flat fills that are one luminance level apart:

| Region | Colour | Luma |
|--------|--------|------|
| Green tunic | `#488041` | 104 |
| Rust boots / pants | `#B0531F` | 105 |
| Orange hair | `#D66A28` | 131 |
| Grey tablet | `#B0B0A0` | 174 |
| Skin / tan fills | `#F5B25D` | 188 |
| Cream ground | `#FEF4DD` | 245 |

A `-colorspace Gray` conversion therefore merges tunic and boots completely and
brings skin to within 14 levels of the tablet. Instead, each pixel is classified
by hue / saturation / value into its design region and mapped onto a ramp of
designed ink coverages:

```
background 0 %   skin 12 %   tablet 26 %   hair 42 %
tunic 58 %       boots 74 %  outline 100 %
```

This guarantees ≥ 14 levels of separation between any two adjacent regions.
Three further points matter for print quality:

1. **The cream ground is forced to paper white.** Printing `#FEF4DD` on a white
   label produces a light toner wash across the whole face of the label.
2. **The class map is median-filtered before the ramp is applied.** Soft
   gradients that cross a class boundary (notably the hair/boots transition at
   the sideburn) otherwise chatter, and reduce to salt-and-pepper at 1 bit.
   Outlines are re-stamped after the filter so thin strokes are never eroded.
3. **The 1-bit variants are supersampled 2× before thresholding**, which
   anti-aliases the strokes and stops thin lines from breaking up.

The outline anti-aliasing window is kept strictly below luma 104, the darkest
flat fill, so the fills themselves stay flat and are never darkened by it.
