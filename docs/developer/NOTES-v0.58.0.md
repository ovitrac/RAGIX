# RAGIX v0.58.0 Release Notes

**Release Date:** 2025-12-12
**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

---

## Overview

Version 0.58.0 focuses on **Partitioner UI polish** — improving the user experience when working with large codebases (1000+ classes). This release addresses usability issues reported during real-world testing on enterprise Java projects.

---

## New Features

### 1. Search Filter with Button

**Problem:** The search filter was triggering on every keystroke, causing performance issues and unexpected behavior with large class lists.

**Solution:**
- Replaced `oninput` with explicit filter button
- Press **Enter** or click **Filter** to apply search
- Click **✕** to clear search filter
- Both graph nodes AND accordion class lists now filter by search term

**Files Changed:**
- `ragix_web/static/index.html` (lines 3671-3673, 6037-6050)

### 2. Accordion Pagination

**Problem:** Accordion sections showed only 100 classes with "...and 6821 more" ellipsis, making it impossible to browse all classes.

**Solution:**
- Paginated class lists with 50 classes per page
- **◀ Less** / **More ▶** navigation buttons
- Page indicator: "1-50 of 6821"
- Each partition section has independent pagination state

**Files Changed:**
- `ragix_web/static/index.html` (lines 5608-5690, 5997-6000, 2143-2168)

### 3. Dynamic Filtered Counts

**Problem:** When search filter was active, accordion titles still showed total counts, not filtered counts.

**Solution:**
- Titles now show: "MSG-HUB (150 of 6821 classes)" when search is active
- Reverts to "MSG-HUB (6821 classes)" when search is cleared
- Updates dynamically as search term changes

**Files Changed:**
- `ragix_web/static/index.html` (lines 5584, 5643-5651)

### 4. Config Panel Toggle

**Problem:** The configuration panel took significant horizontal space, leaving less room for the graph visualization.

**Solution:**
- Toggle button (◀/▶) on the right edge of config panel
- Click to collapse/expand panel
- When collapsed, content is hidden but toggle button remains visible
- Maximizes graph area for better visualization

**Files Changed:**
- `ragix_web/static/index.html` (lines 1782-1818, 3512, 6515-6520)

### 5. Labels Checkbox Fix

**Problem:** The "Labels (auto on zoom)" checkbox had inverted logic — unchecked showed labels on zoom, checked showed labels always.

**Solution:**
- Checkbox is now **checked by default**
- Checked = Auto mode (labels appear only when zoomed > 1.2)
- Unchecked = Labels always visible

**Files Changed:**
- `ragix_web/static/index.html` (lines 3720, 6102-6122)

### 6. Fullscreen Layout Fix

**Problem:** Fullscreen mode had gaps at the bottom, not filling the entire viewport.

**Solution:**
- Changed from fixed height `calc(100vh - 120px)` to flexbox `flex: 1`
- Fullscreen mode now uses `position: fixed` with full viewport coverage
- Added minimal styling for author footer to reduce space usage

**Files Changed:**
- `ragix_web/static/index.html` (lines 1733-1750, 2181-2203)

---

## Technical Details

### New JavaScript Methods

```javascript
// Search filter
applySearchFilter()    // Apply search from input field
clearSearchFilter()    // Clear search and reset

// Accordion pagination
renderPartitionPage(partitionLabel)        // Render page with filtering
changePartitionPage(partitionLabel, delta) // Navigate pages
```

### New State Variables

```javascript
classesPerPage: 50,           // Classes per accordion page
partitionClasses: {},         // Full class data by partition
partitionPages: {},           // Current page per partition
partitionColorMap: {},        // Cached color map
```

### New CSS Classes

```css
.partition-pagination { }      // Pagination container
.pagination-controls { }       // Button row
.pagination-info { }           // Page counter
.author-footer { }             // Minimal footer styling
.partitioner-panel.fullscreen { }  // Fullscreen mode
```

---

## Migration Notes

No breaking changes. All existing configurations and saved states remain compatible.

---

## Dependencies

No new dependencies added.

---

## Known Issues

None reported.

---

## What's Next (v0.59.0)

Potential areas for future improvement:
- Virtual scrolling for extremely large class lists (10K+)
- Keyboard shortcuts for pagination
- Export filtered results
- Search history/suggestions

---

## Contributors

- **Olivier Vitrac** — Architecture, implementation, testing
