// Moteur du presentateur — fichier partage
// Requiert : slide_files (Array), PRESENTER_CHANNEL (String), PRESENTER_NOTES_PREFIX (String) definis avant chargement
(function() {
  var channel_name = window.PRESENTER_CHANNEL || 'slide_presenter';
  var notes_prefix = window.PRESENTER_NOTES_PREFIX || 'presenter_notes_';

  var current_index = parseInt(window.location.hash.replace('#', '')) || 0;
  var presentation_window = null;
  var timer_seconds = 0;
  var timer_interval = null;

  var channel = new BroadcastChannel(channel_name);

  // Elements
  var current_frame = document.getElementById('current-frame');
  var next_frame = document.getElementById('next-frame');
  var slide_counter = document.getElementById('slide-counter');
  var notes_area = document.getElementById('notes-area');
  var timer_display = document.getElementById('timer');
  var end_label = document.getElementById('end-label');
  var current_wrapper = document.getElementById('current-wrapper');
  var btn_lamp = document.getElementById('btn-lamp');
  var btn_laser = document.getElementById('btn-laser');

  // Navigation
  function go_to(index) {
    if (index < 0 || index >= slide_files.length) return;
    save_notes();
    current_index = index;
    window.location.hash = '#' + current_index;
    update_view();
    channel.postMessage({ type: 'navigate', slide: slide_files[current_index] });
  }

  function update_view() {
    current_frame.src = slide_files[current_index];
    slide_counter.textContent = (current_index + 1) + ' / ' + slide_files.length;

    if (current_index + 1 < slide_files.length) {
      next_frame.src = slide_files[current_index + 1];
      next_frame.style.display = '';
      end_label.style.display = 'none';
    } else {
      next_frame.src = '';
      next_frame.style.display = 'none';
      end_label.style.display = '';
    }

    load_notes();
    scale_all();
  }

  // Mise a l'echelle des iframes
  var current_scale = 1;
  var current_offset_x = 0;
  var current_offset_y = 0;

  function scale_iframe(wrapper, iframe) {
    var w = wrapper.clientWidth;
    var h = wrapper.clientHeight;
    if (w === 0 || h === 0) return;
    var scale = Math.min(w / 1280, h / 720);
    iframe.style.transform = 'scale(' + scale + ')';
    var sw = 1280 * scale;
    var sh = 720 * scale;
    var ox = (w - sw) / 2;
    var oy = (h - sh) / 2;
    iframe.style.left = ox + 'px';
    iframe.style.top = oy + 'px';
    if (wrapper.id === 'current-wrapper') {
      current_scale = scale;
      current_offset_x = ox;
      current_offset_y = oy;
    }
  }

  function scale_all() {
    scale_iframe(document.getElementById('current-wrapper'), current_frame);
    scale_iframe(document.getElementById('next-wrapper'), next_frame);
  }

  // Notes (localStorage)
  function notes_key(index) {
    return notes_prefix + slide_files[index];
  }

  function save_notes() {
    localStorage.setItem(notes_key(current_index), notes_area.value);
  }

  function load_notes() {
    var saved = localStorage.getItem(notes_key(current_index));
    var defaults = window.SLIDE_NOTES || {};
    var default_note = defaults[slide_files[current_index]] || '';
    if (saved && saved !== default_note) {
      notes_area.value = saved;
    } else {
      notes_area.value = default_note;
    }
  }

  notes_area.addEventListener('input', function() {
    save_notes();
  });

  // Timer
  function format_time(s) {
    var h = Math.floor(s / 3600);
    var m = Math.floor((s % 3600) / 60);
    var sec = s % 60;
    return String(h).padStart(2, '0') + ':' +
           String(m).padStart(2, '0') + ':' +
           String(sec).padStart(2, '0');
  }

  function start_timer() {
    if (timer_interval) return;
    timer_interval = setInterval(function() {
      timer_seconds++;
      timer_display.textContent = format_time(timer_seconds);
    }, 1000);
  }

  function reset_timer() {
    timer_seconds = 0;
    timer_display.textContent = '00:00:00';
  }

  // Fenetre de presentation
  function open_presentation() {
    presentation_window = window.open(
      slide_files[current_index],
      'slide_presentation'
    );
    start_timer();
  }

  // Boutons
  document.getElementById('btn-prev').addEventListener('click', function() {
    go_to(current_index - 1);
  });

  document.getElementById('btn-next').addEventListener('click', function() {
    go_to(current_index + 1);
  });

  document.getElementById('btn-present').addEventListener('click', function() {
    open_presentation();
  });

  document.getElementById('btn-fullscreen').addEventListener('click', function() {
    channel.postMessage({ type: 'fullscreen' });
  });

  document.getElementById('btn-fullscreen-self').addEventListener('click', function() {
    if (document.fullscreenElement) {
      document.exitFullscreen();
    } else {
      document.documentElement.requestFullscreen().catch(function() {});
    }
  });

  document.getElementById('btn-reset').addEventListener('click', function() {
    reset_timer();
  });

  document.getElementById('btn-exit').addEventListener('click', function() {
    if (presentation_window && !presentation_window.closed) {
      presentation_window.close();
    }
    window.location.href = slide_files[current_index];
  });

  // Mode lampe
  var lamp_slide_radius = 140;
  var lamp_active = false;
  var lamp_overlay = null;

  function toggle_lamp() {
    lamp_active = !lamp_active;
    if (lamp_active) {
      lamp_overlay = document.createElement('div');
      lamp_overlay.id = 'lamp-overlay';
      current_wrapper.appendChild(lamp_overlay);
      current_wrapper.classList.add('lamp-active');
      btn_lamp.classList.add('active');
      channel.postMessage({ type: 'lamp_on' });
    } else {
      if (lamp_overlay) {
        lamp_overlay.remove();
        lamp_overlay = null;
      }
      current_wrapper.classList.remove('lamp-active');
      btn_lamp.classList.remove('active');
      channel.postMessage({ type: 'lamp_off' });
    }
  }

  // Pointeur laser
  var laser_active = false;
  var laser_dot = null;

  function toggle_laser() {
    laser_active = !laser_active;
    if (laser_active) {
      if (lamp_active) toggle_lamp();
      laser_dot = document.createElement('div');
      laser_dot.id = 'laser-dot-local';
      current_wrapper.appendChild(laser_dot);
      current_wrapper.classList.add('lamp-active');
      btn_laser.classList.add('active-laser');
      channel.postMessage({ type: 'laser_on' });
    } else {
      if (laser_dot) {
        laser_dot.remove();
        laser_dot = null;
      }
      current_wrapper.classList.remove('lamp-active');
      btn_laser.classList.remove('active-laser');
      channel.postMessage({ type: 'laser_off' });
    }
  }

  // Dernieres positions connues pour resynchronisation apres navigation
  var last_slide_x = 640;
  var last_slide_y = 360;

  // Mousemove partage (lampe + laser)
  current_wrapper.addEventListener('mousemove', function(e) {
    if (!lamp_active && !laser_active) return;
    var rect = current_wrapper.getBoundingClientRect();
    var mx = e.clientX - rect.left;
    var my = e.clientY - rect.top;

    var sx = (mx - current_offset_x) / current_scale;
    var sy = (my - current_offset_y) / current_scale;

    last_slide_x = sx;
    last_slide_y = sy;

    if (lamp_active && lamp_overlay) {
      var local_radius = lamp_slide_radius * current_scale;
      lamp_overlay.style.background =
        'radial-gradient(circle ' + local_radius + 'px at ' + mx + 'px ' + my + 'px, ' +
        'transparent 0%, transparent 80%, rgba(0,0,0,0.85) 100%)';
      channel.postMessage({ type: 'lamp_move', x: sx, y: sy });
    }

    if (laser_active && laser_dot) {
      laser_dot.style.left = mx + 'px';
      laser_dot.style.top = my + 'px';
      channel.postMessage({ type: 'laser_move', x: sx, y: sy });
    }
  });

  btn_lamp.addEventListener('click', function() {
    toggle_lamp();
  });

  btn_laser.addEventListener('click', function() {
    toggle_laser();
  });

  // Ecouter les messages retour depuis la page presentee
  channel.addEventListener('message', function(e) {
    if (e.data.type === 'audience_navigate') {
      save_notes();
      current_index = e.data.index;
      window.location.hash = '#' + current_index;
      update_view();
    }
    if (e.data.type === 'slide_ready') {
      if (lamp_active) {
        channel.postMessage({ type: 'lamp_on' });
        channel.postMessage({ type: 'lamp_move', x: last_slide_x, y: last_slide_y });
      }
      if (laser_active) {
        channel.postMessage({ type: 'laser_on' });
        channel.postMessage({ type: 'laser_move', x: last_slide_x, y: last_slide_y });
      }
    }
    if (e.data.type === 'audience_fullscreen') {
      if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen().catch(function() {});
      }
    }
    if (e.data.type === 'audience_exit_fullscreen') {
      if (document.fullscreenElement) {
        document.exitFullscreen();
      }
    }
  });

  // Raccourcis clavier
  document.addEventListener('keydown', function(e) {
    if (e.target === notes_area) return;

    if (e.key === 'f' || e.key === 'F') {
      e.preventDefault();
      if (document.fullscreenElement) {
        document.exitFullscreen();
      } else {
        document.documentElement.requestFullscreen().catch(function() {});
      }
      return;
    }
    if (e.key === 'z' || e.key === 'Z') {
      e.preventDefault();
      if (laser_active) toggle_laser();
      toggle_lamp();
      return;
    }
    if (e.key === 'a' || e.key === 'A') {
      e.preventDefault();
      if (lamp_active) toggle_lamp();
      toggle_laser();
      return;
    }
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
      e.preventDefault();
      go_to(current_index + 1);
    }
    if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
      e.preventDefault();
      go_to(current_index - 1);
    }
    if (e.key === 'Home') {
      e.preventDefault();
      go_to(0);
    }
    if (e.key === 'End') {
      e.preventDefault();
      go_to(slide_files.length - 1);
    }
    if (e.key === 'Escape') {
      e.preventDefault();
      if (lamp_active) {
        toggle_lamp();
      } else if (laser_active) {
        toggle_laser();
      } else if (document.fullscreenElement) {
        document.exitFullscreen();
      } else {
        document.getElementById('btn-exit').click();
      }
      return;
    }
  });

  // Splitter redimensionnable
  var splitter = document.getElementById('splitter');
  var presenter_el = document.getElementById('presenter');
  var splitter_dragging = false;

  var stored_ratio = localStorage.getItem('presenter_split_ratio');
  if (stored_ratio) {
    var ratio = parseFloat(stored_ratio);
    presenter_el.style.setProperty('--col-left', ratio + 'fr');
    presenter_el.style.setProperty('--col-right', (1 - ratio) + 'fr');
  }

  function disable_iframes() {
    current_frame.style.pointerEvents = 'none';
    next_frame.style.pointerEvents = 'none';
  }

  function enable_iframes() {
    current_frame.style.pointerEvents = '';
    next_frame.style.pointerEvents = '';
  }

  splitter.addEventListener('mousedown', function(e) {
    e.preventDefault();
    splitter_dragging = true;
    splitter.classList.add('dragging');
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
    disable_iframes();
  });

  document.addEventListener('mousemove', function(e) {
    if (!splitter_dragging) return;
    var rect = presenter_el.getBoundingClientRect();
    var padding = 12;
    var splitter_width = 6;
    var available = rect.width - padding * 2 - splitter_width;
    var x = e.clientX - rect.left - padding;
    var ratio = Math.max(0.3, Math.min(0.8, x / available));
    presenter_el.style.setProperty('--col-left', ratio + 'fr');
    presenter_el.style.setProperty('--col-right', (1 - ratio) + 'fr');
    scale_all();
  });

  document.addEventListener('mouseup', function() {
    if (!splitter_dragging) return;
    splitter_dragging = false;
    splitter.classList.remove('dragging');
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
    enable_iframes();
    var style = getComputedStyle(presenter_el);
    var cols = style.gridTemplateColumns.split(' ');
    var left_px = parseFloat(cols[0]);
    var right_px = parseFloat(cols[2]);
    var ratio = left_px / (left_px + right_px);
    localStorage.setItem('presenter_split_ratio', ratio.toFixed(3));
    scale_all();
  });

  // Splitter horizontal (next / notes)
  var splitter_h = document.getElementById('splitter-h');
  var right_panel = document.getElementById('right-panel');
  var next_wrapper = document.getElementById('next-wrapper');
  var splitter_h_dragging = false;

  if (splitter_h) {
    var stored_h_ratio = localStorage.getItem('presenter_split_h_ratio');
    if (stored_h_ratio) {
      next_wrapper.style.height = parseFloat(stored_h_ratio) * 100 + '%';
    }

    splitter_h.addEventListener('mousedown', function(e) {
      e.preventDefault();
      splitter_h_dragging = true;
      splitter_h.classList.add('dragging');
      document.body.style.cursor = 'row-resize';
      document.body.style.userSelect = 'none';
      disable_iframes();
    });

    document.addEventListener('mousemove', function(e) {
      if (!splitter_h_dragging) return;
      var rect = right_panel.getBoundingClientRect();
      var y = e.clientY - rect.top;
      var ratio = Math.max(0.15, Math.min(0.7, y / rect.height));
      next_wrapper.style.height = ratio * 100 + '%';
      scale_all();
    });

    document.addEventListener('mouseup', function() {
      if (!splitter_h_dragging) return;
      splitter_h_dragging = false;
      splitter_h.classList.remove('dragging');
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      enable_iframes();
      var rect = right_panel.getBoundingClientRect();
      var ratio = next_wrapper.offsetHeight / rect.height;
      localStorage.setItem('presenter_split_h_ratio', ratio.toFixed(3));
      scale_all();
    });
  }

  // Redimensionnement
  window.addEventListener('resize', scale_all);
  new ResizeObserver(scale_all).observe(document.getElementById('current-wrapper'));
  new ResizeObserver(scale_all).observe(document.getElementById('next-wrapper'));

  // Recevoir les touches relayees depuis l'iframe courante
  window.addEventListener('message', function(e) {
    if (e.data && e.data.type === 'relay_keydown') {
      document.dispatchEvent(new KeyboardEvent('keydown', {
        key: e.data.key, code: e.data.code, bubbles: true
      }));
    }
  });

  // Init
  update_view();
  start_timer();
})();
