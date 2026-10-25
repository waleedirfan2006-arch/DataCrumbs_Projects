const express = require('express');
const path = require('path');
const bodyParser = require('body-parser');
const { PythonShell } = require('python-shell');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(bodyParser.json());
app.use(express.static(path.join(__dirname, 'public')));

// Serve index.html
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Predict endpoint – forwards JSON payload to the Python pipeline
app.post('/predict', (req, res) => {
  const input = req.body; // should be an object with feature keys
  const options = {
    mode: 'json',
    pythonOptions: ['-u'], // unbuffered stdout
    scriptPath: path.join(__dirname, 'model'),
    // Use the Windows launcher if `python` isn’t on PATH
    pythonPath: 'py',
  };

  const pyshell = new PythonShell('predict.py', options);
  pyshell.send(JSON.stringify(input));

  let prediction = null;
  pyshell.on('message', (msg) => {
    // Expected output is JSON with a `prediction` field
    if (msg && typeof msg === 'object' && 'prediction' in msg) {
      prediction = msg.prediction;
    }
  });

  pyshell.end((err) => {
    if (err) {
      console.error('Python error:', err);
      return res.status(500).json({ error: 'Model inference failed' });
    }
    if (prediction === null) {
      return res.status(500).json({ error: 'No prediction returned' });
    }
    res.json({ prediction });
  });
});

app.listen(PORT, () => {
  console.log(`Server listening on http://localhost:${PORT}`);
});
