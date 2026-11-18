// Moteur du viewer — fichier partage
// Requiert : VIEWER_SLIDES (Array) et VIEWER_FIT_KEY (String) definis avant chargement
(function() {
  var slides = window.VIEWER_SLIDES;
  var fit_key = window.VIEWER_FIT_KEY || 'wiem-slides-fit';

  var current_index = 0;

  var iframe = document.getElementById('slide-iframe');
  iframe.style.pointerEvents = 'none';
  var current_slide_el = document.getElementById('current-slide');
  var total_slides_el = document.getElementById('total-slides');
  var prev_btn = document.getElementById('prev-btn');
  var next_btn = document.getElementById('next-btn');
  var fullscreen_btn = document.getElementById('fullscreen-btn');
  var thumbnails_container = document.getElementById('thumbnails');
  var fs_current = document.getElementById('fs-current');
  var fs_total = document.getElementById('fs-total');
  var fit_cb = document.getElementById('fit-cb');

  total_slides_el.textContent = slides.length;
  fs_total.textContent = slides.length;
  create_thumbnails();

  var initial_slide = get_slide_from_hash();
  if (initial_slide !== null) {
    go_to_slide(initial_slide);
  } else {
    update_navigation();
  }

  function get_slide_from_hash() {
    var hash = window.location.hash;
    if (hash) {
      var slide_num = parseInt(hash.replace('#', ''), 10);
      if (!isNaN(slide_num) && slide_num >= 1 && slide_num <= slides.length) {
        return slide_num - 1;
      }
    }
    return null;
  }

  function update_hash(index) {
    window.history.replaceState(null, null, '#' + (index + 1));
  }

  window.addEventListener('hashchange', function() {
    var slide_index = get_slide_from_hash();
    if (slide_index !== null && slide_index !== current_index) {
      go_to_slide(slide_index);
    }
  });

  function create_thumbnails() {
    slides.forEach(function(slide, index) {
      var thumb = document.createElement('div');
      thumb.className = 'thumbnail' + (index === 0 ? ' active' : '');
      thumb.innerHTML = '<iframe src="' + slide.file + '" scrolling="no"></iframe>' +
        '<span class="thumbnail-number">' + (index + 1) + '</span>';
      thumb.addEventListener('click', function() { go_to_slide(index); });
      thumbnails_container.appendChild(thumb);
    });
  }

  function go_to_slide(index) {
    if (index < 0 || index >= slides.length) return;
    current_index = index;
    iframe.src = slides[index].file;
    current_slide_el.textContent = index + 1;
    fs_current.textContent = index + 1;
    document.querySelectorAll('.thumbnail').forEach(function(thumb, i) {
      thumb.classList.toggle('active', i === index);
    });
    update_hash(index);
    update_navigation();
  }

  function update_navigation() {
    prev_btn.disabled = current_index === 0;
    next_btn.disabled = current_index === slides.length - 1;
  }

  function next_slide() {
    if (current_index < slides.length - 1) go_to_slide(current_index + 1);
  }

  function prev_slide() {
    if (current_index > 0) go_to_slide(current_index - 1);
  }

  function toggle_fullscreen() {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().catch(function() {});
    } else {
      document.exitFullscreen();
    }
  }

  // Autoplay
  var autoplay_btn = document.getElementById('autoplay-btn');
  var autoplay_icon = document.getElementById('autoplay-icon');
  var autoplay_delay_input = document.getElementById('autoplay-delay');
  var autoplay_timer = null;
  var autoplay_active = false;

  function get_autoplay_delay() {
    return Math.max(1, parseInt(autoplay_delay_input.value) || 5) * 1000;
  }

  function start_autoplay() {
    autoplay_active = true;
    autoplay_btn.classList.add('playing');
    autoplay_icon.className = 'ri-pause-line';
    schedule_autoplay();
  }

  function schedule_autoplay() {
    if (autoplay_timer) clearTimeout(autoplay_timer);
    autoplay_timer = setTimeout(function() {
      if (current_index < slides.length - 1) {
        go_to_slide(current_index + 1);
        schedule_autoplay();
      } else {
        stop_autoplay();
      }
    }, get_autoplay_delay());
  }

  function stop_autoplay() {
    autoplay_active = false;
    autoplay_btn.classList.remove('playing');
    autoplay_icon.className = 'ri-play-line';
    if (autoplay_timer) { clearTimeout(autoplay_timer); autoplay_timer = null; }
  }

  autoplay_btn.addEventListener('click', function() {
    if (autoplay_active) stop_autoplay(); else start_autoplay();
  });

  autoplay_delay_input.addEventListener('change', function() {
    if (autoplay_active) schedule_autoplay();
  });

  autoplay_delay_input.addEventListener('keydown', function(e) { e.stopPropagation(); });

  // Boutons navigation
  prev_btn.addEventListener('click', function() { stop_autoplay(); prev_slide(); });
  next_btn.addEventListener('click', function() { stop_autoplay(); next_slide(); });
  fullscreen_btn.addEventListener('click', toggle_fullscreen);

  // Lampe et laser - relais vers l'iframe
  var lamp_active = false;
  var laser_active = false;

  function send_to_slide(msg) {
    if (iframe.contentWindow) iframe.contentWindow.postMessage(msg, '*');
  }

  iframe.addEventListener('load', function() {
    if (lamp_active) send_to_slide({ type: 'lamp_on' });
    if (laser_active) send_to_slide({ type: 'laser_on' });
  });

  // Relais mousemove vers l'iframe pour lampe/laser
  var slide_frame = document.getElementById('slide-frame');
  document.addEventListener('mousemove', function(e) {
    if (!lamp_active && !laser_active) return;
    if (!slide_frame) return;
    var rect = slide_frame.getBoundingClientRect();
    var sx = (e.clientX - rect.left) / (rect.width / 1280);
    var sy = (e.clientY - rect.top) / (rect.height / 720);
    if (lamp_active) send_to_slide({ type: 'lamp_move', x: sx, y: sy });
    if (laser_active) send_to_slide({ type: 'laser_move', x: sx, y: sy });
  });

  // Raccourcis clavier
  document.addEventListener('keydown', function(e) {
    switch (e.key) {
      case 'ArrowLeft': stop_autoplay(); prev_slide(); break;
      case 'ArrowRight': case ' ': stop_autoplay(); next_slide(); e.preventDefault(); break;
      case 'f': case 'F': toggle_fullscreen(); break;
      case 'w': case 'W': fit_cb.checked = !fit_cb.checked; fit_cb.dispatchEvent(new Event('change')); break;
      case 'p': case 'P':
        var folder = slides[0].file.split('/').slice(0, -1).join('/');
        window.location.href = folder + '/presenter.html#' + current_index;
        break;
      case 'z': case 'Z':
        lamp_active = !lamp_active;
        if (lamp_active) {
          if (laser_active) { laser_active = false; send_to_slide({ type: 'laser_off' }); }
          send_to_slide({ type: 'lamp_on' });
        } else {
          send_to_slide({ type: 'lamp_off' });
        }
        break;
      case 'a': case 'A':
        laser_active = !laser_active;
        if (laser_active) {
          if (lamp_active) { lamp_active = false; send_to_slide({ type: 'lamp_off' }); }
          send_to_slide({ type: 'laser_on' });
        } else {
          send_to_slide({ type: 'laser_off' });
        }
        break;
      case 'Escape':
        if (lamp_active) { lamp_active = false; send_to_slide({ type: 'lamp_off' }); }
        else if (laser_active) { laser_active = false; send_to_slide({ type: 'laser_off' }); }
        else if (document.fullscreenElement) document.exitFullscreen();
        break;
      case 'Home': go_to_slide(0); break;
      case 'End': go_to_slide(slides.length - 1); break;
    }
  });

  // Fullscreen
  document.addEventListener('fullscreenchange', function() {
    if (document.fullscreenElement) {
      document.body.classList.add('is-fullscreen');
      fullscreen_btn.innerHTML = '<i class="ri-fullscreen-exit-line"></i>';
    } else {
      document.body.classList.remove('is-fullscreen');
      fullscreen_btn.innerHTML = '<i class="ri-fullscreen-line"></i>';
    }
    scale_slide();
  });

  // Fit mode
  var stored_fit = localStorage.getItem(fit_key);
  var is_fit = stored_fit === null ? true : stored_fit === '1';
  fit_cb.checked = is_fit;
  if (is_fit) document.getElementById('thumbnails-bar').classList.add('hidden');

  fit_cb.addEventListener('change', function() {
    is_fit = this.checked;
    localStorage.setItem(fit_key, is_fit ? '1' : '0');
    document.getElementById('thumbnails-bar').classList.toggle('hidden', is_fit);
    scale_slide();
  });

  // Mise a l'echelle
  function scale_slide() {
    var container = document.getElementById('slide-container');
    var frame = document.getElementById('slide-frame');
    var is_fullscreen = document.body.classList.contains('is-fullscreen');
    var slide_width = 1280;
    var slide_height = 720;
    var available_width, available_height;
    if (is_fullscreen) {
      available_width = window.innerWidth;
      available_height = window.innerHeight;
    } else if (is_fit) {
      available_width = container.clientWidth - 16;
      available_height = container.clientHeight - 8;
    } else {
      available_width = container.clientWidth - 160;
      available_height = container.clientHeight - 80;
    }
    var scale_x = available_width / slide_width;
    var scale_y = available_height / slide_height;
    var scale = Math.min(scale_x, scale_y);
    frame.style.transform = 'scale(' + scale + ')';
  }

  window.addEventListener('resize', scale_slide);
  new ResizeObserver(scale_slide).observe(document.getElementById('slide-container'));
  scale_slide();

  // Empecher l'iframe de capturer le focus (les raccourcis doivent rester sur le viewer)
  iframe.addEventListener('focus', function() { this.blur(); window.focus(); });
  document.addEventListener('visibilitychange', function() {
    if (!document.hidden) window.focus();
  });
  window.addEventListener('focus', function() {
    if (document.activeElement === iframe) iframe.blur();
  });
  document.getElementById('slide-container').addEventListener('click', function() {
    window.focus();
  });

  // Recevoir les touches relayees depuis l'iframe (quand l'iframe a le focus)
  window.addEventListener('message', function(e) {
    if (e.data && e.data.type === 'relay_keydown') {
      document.dispatchEvent(new KeyboardEvent('keydown', { key: e.data.key, code: e.data.code, bubbles: true }));
    }
  });
})();
