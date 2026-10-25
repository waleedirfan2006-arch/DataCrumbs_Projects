document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('predict-form');
  const resultDiv = document.getElementById('result');

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData(form);
    const payload = {};
    formData.forEach((value, key) => {
      // Convert numeric fields to numbers
      const num = Number(value);
      payload[key] = isNaN(num) ? value : num;
    });

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (response.ok) {
        resultDiv.textContent = `💰 Predicted price: $${Number(data.prediction).toLocaleString(undefined, { maximumFractionDigits: 2 })}`;
      } else {
        resultDiv.textContent = `Error: ${data.error || 'Unknown error'}`;
      }
    } catch (err) {
      resultDiv.textContent = `Network error: ${err.message}`;
    }
  });
});
