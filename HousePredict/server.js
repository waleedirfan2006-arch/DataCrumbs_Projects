const express = require('express');
const cors = require('cors');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// Load Model Parameters and Market Stats
const modelParamsPath = path.join(__dirname, 'model', 'model_params.json');
const cityStatsPath = path.join(__dirname, 'model', 'city_stats.json');

let model = null;
let cityStats = {};

try {
  if (fs.existsSync(modelParamsPath)) {
    model = JSON.parse(fs.readFileSync(modelParamsPath, 'utf8'));
    console.log('Loaded model_params.json successfully.');
  }
  if (fs.existsSync(cityStatsPath)) {
    cityStats = JSON.parse(fs.readFileSync(cityStatsPath, 'utf8'));
    console.log(`Loaded city statistics for ${Object.keys(cityStats).length} cities.`);
  }
} catch (err) {
  console.error('Error loading model or stats:', err);
}

// Prediction function using scikit-learn pipeline exact weights
function calculatePrice(input) {
  if (!model) throw new Error('Model is not initialized.');

  // Numerical features scaled via (val - mean) / scale
  const scaledNums = model.num_features.map((feature, idx) => {
    const val = Number(input[feature] ?? model.num_means[idx]);
    return (val - model.num_means[idx]) / model.num_scales[idx];
  });

  // Categorical One-Hot Encoding for city
  const encodedCat = model.categories.map(cat => (input.city === cat ? 1.0 : 0.0));
  const fullFeatureVector = [...scaledNums, ...encodedCat];

  // Dot product with coefficients + intercept
  let predictedPrice = model.intercept;
  for (let i = 0; i < fullFeatureVector.length; i++) {
    predictedPrice += fullFeatureVector[i] * model.coefficients[i];
  }

  // Realistic market minimum threshold
  const finalPrice = Math.max(predictedPrice, 15000);
  return finalPrice;
}

// API Routes
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

app.get('/api/model-info', (req, res) => {
  if (!model) return res.status(500).json({ error: 'Model parameters not loaded' });
  res.json({
    algorithm: 'Multiple Linear Regression (sklearn.linear_model.LinearRegression)',
    target: 'price (USD $)',
    location: 'King County, Washington, USA',
    metrics: model.metrics,
    features: {
      numerical: model.num_features,
      categorical: model.cat_features,
      totalDimensions: model.coefficients.length
    },
    totalListings: 4600,
    activeCitiesCount: Object.keys(cityStats).length
  });
});

app.get('/api/cities', (req, res) => {
  res.json(cityStats);
});

app.post('/api/predict', (req, res) => {
  try {
    const input = req.body || {};

    // Basic validation
    const bedrooms = parseFloat(input.bedrooms) || 3.0;
    const bathrooms = parseFloat(input.bathrooms) || 2.0;
    const sqft_living = parseInt(input.sqft_living) || 2000;
    const sqft_lot = parseInt(input.sqft_lot) || 6000;
    const floors = parseFloat(input.floors) || 1.5;
    const waterfront = parseInt(input.waterfront) ? 1 : 0;
    const view = parseInt(input.view) || 0;
    const condition = parseInt(input.condition) || 3;
    const sqft_above = parseInt(input.sqft_above) || 1500;
    const sqft_basement = parseInt(input.sqft_basement) || 500;
    const yr_built = parseInt(input.yr_built) || 1980;
    const yr_renovated = parseInt(input.yr_renovated) || 0;
    const city = (input.city || 'Seattle').trim();

    const normalizedInput = {
      bedrooms, bathrooms, sqft_living, sqft_lot, floors,
      waterfront, view, condition, sqft_above, sqft_basement,
      yr_built, yr_renovated, city
    };

    const price = calculatePrice(normalizedInput);
    const pricePerSqft = Math.round(price / Math.max(sqft_living, 1));
    const confidenceRange = {
      low: Math.max(price - (model.metrics?.mae || 131458), 10000),
      high: price + (model.metrics?.mae || 131458)
    };

    const cityBenchmark = cityStats[city] || { median: 465000, avgPrice: 550000 };

    res.json({
      price: Math.round(price),
      currency: 'USD',
      pricePerSqft,
      confidenceRange: {
        low: Math.round(confidenceRange.low),
        high: Math.round(confidenceRange.high)
      },
      cityBenchmark,
      inputEvaluated: normalizedInput
    });
  } catch (error) {
    console.error('Prediction error:', error);
    res.status(500).json({ error: error.message || 'Internal error calculating valuation' });
  }
});

// Fallback to index.html for SPA navigation
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.listen(PORT, () => {
  console.log(`RedSun HousePredict AI server running at http://localhost:${PORT}`);
});
