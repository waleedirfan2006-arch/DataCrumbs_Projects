const request = require('supertest');
const express = require('express');
const bodyParser = require('body-parser');
const path = require('path');
const { PythonShell } = require('python-shell');

// Re‑use the app.js logic but expose it as a function for testing
function createApp() {
  const app = express();
  app.use(bodyParser.json());
  app.use(express.static(path.join(__dirname, '..', 'public')));

  app.post('/predict', (req, res) => {
    const input = req.body;
    const options = {
      mode: 'json',
      pythonOptions: ['-u'],
      scriptPath: path.join(__dirname, '..', 'model'),
      pythonPath: 'py',
    };
    const pyshell = new PythonShell('predict.py', options);
    pyshell.send(JSON.stringify(input));
    let prediction = null;
    pyshell.on('message', (msg) => {
      if (msg && typeof msg === 'object' && 'prediction' in msg) {
        prediction = msg.prediction;
      }
    });
    pyshell.end((err) => {
      if (err) return res.status(500).json({ error: 'Model inference failed' });
      if (prediction === null) return res.status(500).json({ error: 'No prediction returned' });
      res.json({ prediction });
    });
  });

  return app;
}

module.exports = { createApp };
