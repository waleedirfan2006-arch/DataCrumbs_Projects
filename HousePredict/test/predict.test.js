const request = require('supertest');
const { createApp } = require('../app_test'); // our testable app

describe('POST /predict', () => {
  test('returns a prediction for valid input', async () => {
    // Minimal valid payload (use typical ranges)
    const payload = {
      date: '2020-01-01',
      bedrooms: 3,
      bathrooms: 2,
      sqft_living: 2000,
      sqft_lot: 5000,
      floors: 1,
      waterfront: 0,
      view: 0,
      condition: 3,
      sqft_above: 1800,
      sqft_basement: 200,
      yr_built: 1990,
      yr_renovated: 0,
      street: '123 Main St',
      city: 'Seattle',
      statezip: 'WA 98101',
      country: 'USA'
    };
    const app = createApp();
    const response = await request(app).post('/predict').send(payload);
    expect(response.status).toBe(200);
    expect(response.body).toHaveProperty('prediction');
    expect(typeof response.body.prediction).toBe('number');
  }, 20000); // increase timeout for python call
});
