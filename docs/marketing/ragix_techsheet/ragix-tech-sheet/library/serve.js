// Serveur HTTP local pour les presentations Adservio
// Necessaire pour BroadcastChannel (mode presentateur) qui ne fonctionne pas en file://
const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = process.argv[2] || 3000;
const BASE = process.cwd();

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.svg': 'image/svg+xml',
  '.gif': 'image/gif',
  '.ico': 'image/x-icon',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2',
  '.ttf': 'font/ttf',
  '.pdf': 'application/pdf'
};

const server = http.createServer(function(req, res) {
  var url_path = decodeURIComponent(req.url.split('?')[0]);
  if (url_path === '/') url_path = '/index.html';

  var file_path = path.join(BASE, url_path);

  // Securite : empecher la traversee de repertoire
  if (!file_path.startsWith(BASE)) {
    res.writeHead(403);
    res.end('Interdit');
    return;
  }

  fs.stat(file_path, function(err, stats) {
    if (err || !stats.isFile()) {
      res.writeHead(404);
      res.end('Non trouve : ' + url_path);
      return;
    }

    var ext = path.extname(file_path).toLowerCase();
    var mime = MIME[ext] || 'application/octet-stream';

    res.writeHead(200, { 'Content-Type': mime });
    fs.createReadStream(file_path).pipe(res);
  });
});

server.listen(PORT, function() {
  console.log('Serveur local demarre sur http://localhost:' + PORT);
  console.log('Racine : ' + BASE);
  console.log('');
  console.log('  Viewer      : http://localhost:' + PORT + '/index.html');
  console.log('  Presentateur: http://localhost:' + PORT + '/library/presenter.html');
  console.log('');
  console.log('Le mode presentateur (touche P) et BroadcastChannel fonctionnent via HTTP.');
  console.log('Ctrl+C pour arreter.');
});
