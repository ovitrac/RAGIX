// Configuration des slides — modifier le tableau ci-dessous
var slide_files = [
  'slide-01-couverture.html',
  'slide-01b-sommaire.html',
  'slide-02-defi.html',
  'slide-03-plateforme.html',
  'slide-04-differenciateurs.html',
  'slide-05-faits-documents.html',
  'slide-06-generation-pipeline.html',
  'slide-07-capacites.html',
  'slide-08-raisonnement.html',
  'slide-09-souverain.html',
  'slide-10-preuve.html',
  'slide-11-cas.html',
  'slide-12-forge.html',
  'slide-13-cloture.html'
];
var SLIDE_CHANNEL = 'slide_presenter';

// Moteur de navigation des slides — fichier partage
// Requiert : slide_files (Array) et SLIDE_CHANNEL (String) definis avant chargement
(function() {
  var channel_name = window.SLIDE_CHANNEL || 'slide_presenter';

  var currentFile = window.location.pathname.split('/').pop();
  var currentIndex = slide_files.indexOf(currentFile);
  var in_iframe = window.parent !== window;
  var presenter_channel = in_iframe ? null : new BroadcastChannel(channel_name);

  // Navigation iframe pour préserver le fullscreen
  var nav_iframe = null;

  function navigate_via_iframe(index) {
    if (!nav_iframe) {
      nav_iframe = document.createElement('iframe');
      nav_iframe.id = 'nav-iframe';
      Object.assign(nav_iframe.style, {
        position: 'fixed',
        width: '1280px', height: '720px',
        border: 'none',
        zIndex: '99999',
        background: '#000',
        transformOrigin: '0 0'
      });
      nav_iframe.addEventListener('load', function() {
        if (lamp_active) nav_iframe.contentWindow.postMessage({ type: 'lamp_on' }, '*');
        if (laser_active) nav_iframe.contentWindow.postMessage({ type: 'laser_on' }, '*');
      });
      document.body.appendChild(nav_iframe);
      if (slide_container) slide_container.style.display = 'none';
      scale_nav_iframe();
      window.addEventListener('resize', scale_nav_iframe);
    }
    nav_iframe.src = slide_files[index];
    currentIndex = index;
    currentFile = slide_files[index];
  }

  function scale_nav_iframe() {
    if (!nav_iframe) return;
    var vw = window.innerWidth;
    var vh = window.innerHeight;
    var scale = Math.min(vw / 1280, vh / 720);
    nav_iframe.style.transform = 'scale(' + scale + ')';
    nav_iframe.style.left = ((vw - 1280 * scale) / 2) + 'px';
    nav_iframe.style.top = ((vh - 720 * scale) / 2) + 'px';
  }

  function cleanup_nav_iframe() {
    if (nav_iframe) {
      nav_iframe.remove();
      nav_iframe = null;
      window.removeEventListener('resize', scale_nav_iframe);
      if (slide_container) slide_container.style.display = '';
    }
  }

  function navigate_to(index) {
    if (document.fullscreenElement || nav_iframe) {
      navigate_via_iframe(index);
    } else {
      window.location.href = slide_files[index];
    }
  }

  function navigate(direction) {
    var newIndex = currentIndex + direction;
    if (newIndex >= 0 && newIndex < slide_files.length) {
      if (presenter_channel) {
        presenter_channel.postMessage({
          type: 'audience_navigate',
          slide: slide_files[newIndex],
          index: newIndex
        });
      }
      navigate_to(newIndex);
    }
  }

  var slide_container = document.querySelector('.container') || document.querySelector('.slide-container');
  if (slide_container) slide_container.style.position = 'relative';

  function client_to_slide(cx, cy) {
    if (!slide_container) return { x: cx, y: cy };
    var rect = slide_container.getBoundingClientRect();
    return {
      x: (cx - rect.left) / (rect.width / 1280),
      y: (cy - rect.top) / (rect.height / 720)
    };
  }

  // Mode lampe (spotlight)
  var lamp_active = false;
  var lamp_overlay = null;
  var lamp_x = 640;
  var lamp_y = 360;
  var lamp_radius = 140;

  function create_lamp() {
    lamp_overlay = document.createElement('div');
    lamp_overlay.id = 'lamp-overlay';
    Object.assign(lamp_overlay.style, {
      position: 'absolute', top: '0', left: '0',
      width: '1280px', height: '720px', zIndex: '9999', pointerEvents: 'none'
    });
    slide_container.appendChild(lamp_overlay);
    update_lamp();
  }

  function update_lamp() {
    if (!lamp_overlay) return;
    lamp_overlay.style.background =
      'radial-gradient(circle ' + lamp_radius + 'px at ' + lamp_x + 'px ' + lamp_y + 'px, ' +
      'transparent 0%, transparent 80%, rgba(0,0,0,0.85) 100%)';
  }

  function toggle_lamp() {
    lamp_active = !lamp_active;
    if (lamp_active) {
      if (laser_active) toggle_laser();
      if (nav_iframe) {
        nav_iframe.contentWindow.postMessage({ type: 'lamp_on' }, '*');
      } else {
        create_lamp();
      }
      document.body.style.cursor = 'none';
    } else {
      if (nav_iframe) {
        nav_iframe.contentWindow.postMessage({ type: 'lamp_off' }, '*');
      } else {
        if (lamp_overlay) { lamp_overlay.remove(); lamp_overlay = null; }
      }
      document.body.style.cursor = '';
    }
  }

  // Pointeur laser
  var laser_active = false;
  var laser_dot = null;
  var laser_x = 640;
  var laser_y = 360;

  function create_laser() {
    laser_dot = document.createElement('div');
    laser_dot.id = 'laser-dot';
    Object.assign(laser_dot.style, {
      position: 'absolute', width: '22px', height: '22px', borderRadius: '50%',
      background: '#ff0000',
      boxShadow: '0 0 12px 6px rgba(255,0,0,0.6), 0 0 30px 12px rgba(255,0,0,0.3)',
      zIndex: '9999', pointerEvents: 'none', transform: 'translate(-50%, -50%)'
    });
    slide_container.appendChild(laser_dot);
    update_laser();
  }

  function update_laser() {
    if (!laser_dot) return;
    laser_dot.style.left = laser_x + 'px';
    laser_dot.style.top = laser_y + 'px';
  }

  function toggle_laser() {
    laser_active = !laser_active;
    if (laser_active) {
      if (lamp_active) toggle_lamp();
      if (nav_iframe) {
        nav_iframe.contentWindow.postMessage({ type: 'laser_on' }, '*');
      } else {
        create_laser();
      }
      document.body.style.cursor = 'none';
    } else {
      if (nav_iframe) {
        nav_iframe.contentWindow.postMessage({ type: 'laser_off' }, '*');
      } else {
        if (laser_dot) { laser_dot.remove(); laser_dot = null; }
      }
      document.body.style.cursor = '';
    }
  }

  // Mousemove
  document.addEventListener('mousemove', function(e) {
    if (!lamp_active && !laser_active) return;
    var pos = client_to_slide(e.clientX, e.clientY);
    if (lamp_active) { lamp_x = pos.x; lamp_y = pos.y; update_lamp(); }
    if (laser_active) { laser_x = pos.x; laser_y = pos.y; update_laser(); }
  });

  // Synchronisation presentateur via BroadcastChannel
  if (presenter_channel) presenter_channel.addEventListener('message', function(e) {
    if (e.data.type === 'navigate') {
      var idx = slide_files.indexOf(e.data.slide);
      if (idx !== -1) navigate_to(idx);
      else window.location.href = e.data.slide;
    }
    if (e.data.type === 'fullscreen') {
      sessionStorage.setItem('slide_fullscreen', '1');
      document.documentElement.requestFullscreen().catch(function() {});
    }
    if (e.data.type === 'lamp_on') {
      if (!lamp_active) { lamp_active = true; if (laser_active) toggle_laser(); create_lamp(); document.body.style.cursor = 'none'; }
    }
    if (e.data.type === 'lamp_off') {
      if (lamp_active) { lamp_active = false; if (lamp_overlay) { lamp_overlay.remove(); lamp_overlay = null; } document.body.style.cursor = ''; }
    }
    if (e.data.type === 'lamp_move') { lamp_x = e.data.x; lamp_y = e.data.y; update_lamp(); }
    if (e.data.type === 'laser_on') {
      if (!laser_active) { laser_active = true; if (lamp_active) toggle_lamp(); create_laser(); document.body.style.cursor = 'none'; }
    }
    if (e.data.type === 'laser_off') {
      if (laser_active) { laser_active = false; if (laser_dot) { laser_dot.remove(); laser_dot = null; } document.body.style.cursor = ''; }
    }
    if (e.data.type === 'laser_move') { laser_x = e.data.x; laser_y = e.data.y; update_laser(); }
  });

  // Mise a l'echelle viewport
  var fullscreen_user_exit = false;

  function apply_viewport_scale() {
    if (in_iframe) return;
    if (!slide_container) return;
    var vw = window.innerWidth;
    var vh = window.innerHeight;
    var scale = Math.min(vw / 1280, vh / 720);
    document.body.style.width = vw + 'px';
    document.body.style.height = vh + 'px';
    document.body.style.display = 'flex';
    document.body.style.alignItems = 'center';
    document.body.style.justifyContent = 'center';
    if (document.fullscreenElement) {
      document.body.style.background = '#000';
    } else {
      document.body.style.removeProperty('background');
    }
    slide_container.style.transform = 'scale(' + scale + ')';
    slide_container.style.transformOrigin = 'center center';
    slide_container.style.flexShrink = '0';
  }

  if (!in_iframe) {
    apply_viewport_scale();
    window.addEventListener('resize', apply_viewport_scale);
  }

  // Fullscreen
  document.addEventListener('fullscreenchange', function() {
    apply_viewport_scale();
    if (document.fullscreenElement) {
      sessionStorage.setItem('slide_fullscreen', '1');
      if (presenter_channel) presenter_channel.postMessage({ type: 'audience_fullscreen' });
    } else {
      // Sortie du fullscreen : nettoyer l'iframe de navigation et aller sur la slide courante
      if (nav_iframe) {
        var target = currentFile;
        cleanup_nav_iframe();
        window.location.href = target;
        return;
      }
      if (fullscreen_user_exit) {
        sessionStorage.removeItem('slide_fullscreen');
        fullscreen_user_exit = false;
        if (presenter_channel) presenter_channel.postMessage({ type: 'audience_exit_fullscreen' });
      }
    }
  });

  if (!in_iframe && sessionStorage.getItem('slide_fullscreen') === '1') {
    document.documentElement.requestFullscreen().catch(function() {});
  }

  // Raccourcis clavier
  document.addEventListener('keydown', function(e) {
    // Dans un iframe : relayer les touches de navigation vers le parent
    if (in_iframe) {
      var nav_keys = ['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End', 'Escape', ' '];
      var shortcut_keys = ['f', 'F', 'z', 'Z', 'a', 'A', 'p', 'P', 'w', 'W'];
      if (nav_keys.indexOf(e.key) !== -1 || shortcut_keys.indexOf(e.key) !== -1) {
        e.preventDefault();
        window.parent.postMessage({ type: 'relay_keydown', key: e.key, code: e.code }, '*');
      }
      return;
    }
    if (e.key === 'Escape') {
      if (lamp_active) { e.preventDefault(); toggle_lamp(); }
      else if (laser_active) { e.preventDefault(); toggle_laser(); }
      else if (document.fullscreenElement) {
        e.preventDefault();
        fullscreen_user_exit = true;
        document.exitFullscreen();
      }
      return;
    }
    if (e.key === 'f' || e.key === 'F') {
      if (document.fullscreenElement) { fullscreen_user_exit = true; document.exitFullscreen(); }
      else { document.documentElement.requestFullscreen().catch(function() {}); }
      return;
    }
    if (e.key === 'z' || e.key === 'Z') { toggle_lamp(); return; }
    if (e.key === 'a' || e.key === 'A') { toggle_laser(); return; }
    if (e.key === 'p' || e.key === 'P') { window.location.href = '../library/presenter.html#' + currentIndex; return; }
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') navigate(1);
    if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') navigate(-1);
    if (e.key === 'Home') {
      if (presenter_channel) presenter_channel.postMessage({ type: 'audience_navigate', slide: slide_files[0], index: 0 });
      navigate_to(0);
    }
    if (e.key === 'End') {
      var last = slide_files.length - 1;
      if (presenter_channel) presenter_channel.postMessage({ type: 'audience_navigate', slide: slide_files[last], index: last });
      navigate_to(last);
    }
  });

  if (presenter_channel) {
    presenter_channel.postMessage({ type: 'slide_ready', index: currentIndex });
  }

  // Gestion du bfcache (back/forward cache du navigateur)
  window.addEventListener('pageshow', function(e) {
    if (e.persisted) {
      currentFile = window.location.pathname.split('/').pop();
      currentIndex = slide_files.indexOf(currentFile);
      if (!in_iframe) {
        if (presenter_channel) { try { presenter_channel.close(); } catch(err) {} }
        presenter_channel = new BroadcastChannel(channel_name);
        presenter_channel.postMessage({ type: 'slide_ready', index: currentIndex });
      }
      apply_viewport_scale();
    }
  });

  // Recevoir les touches relayees depuis le nav_iframe
  if (!in_iframe) {
    window.addEventListener('message', function(e) {
      if (e.data.type === 'relay_keydown') {
        document.dispatchEvent(new KeyboardEvent('keydown', { key: e.data.key, code: e.data.code, bubbles: true }));
      }
    });
  }

  // Écouter les messages du viewer parent quand dans un iframe
  if (in_iframe) {
    window.addEventListener('message', function(e) {
      if (e.data.type === 'lamp_on') { if (!lamp_active) toggle_lamp(); }
      if (e.data.type === 'lamp_off') { if (lamp_active) toggle_lamp(); }
      if (e.data.type === 'lamp_move') { lamp_x = e.data.x; lamp_y = e.data.y; update_lamp(); }
      if (e.data.type === 'laser_on') { if (!laser_active) toggle_laser(); }
      if (e.data.type === 'laser_off') { if (laser_active) toggle_laser(); }
      if (e.data.type === 'laser_move') { laser_x = e.data.x; laser_y = e.data.y; update_laser(); }
    });
  }
})();
