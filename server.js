const express = require('express');
const axios = require('axios');
const cors = require('cors');

const app = express();

// Middleware
app.use(cors());
app.use(express.json());

// Hugging Face Space API URL
const HF_API_URL = 'https://pcam-velox.hf.space/run/predict';

// Root route (GET)
app.get('/', (req, res) => {
  res.json({ message: 'VeloxG Proxy Server is live!' });
});

// Simple search route (GET)
app.get('/search', (req, res) => {
  const query = req.query.q || 'No query given';
  res.json({ message: `You searched for: ${query}` });
});

// Proxy endpoint to Hugging Face Space API (POST)
app.post('/predict', async (req, res) => {
  try {
    const response = await axios.post(HF_API_URL, req.body, {
      headers: { 'Content-Type': 'application/json' }
    });
    res.json(response.data);
  } catch (error) {
    console.error('Error forwarding request to Hugging Face:', error.message);
    res.status(500).json({
      error: 'Failed to fetch prediction from Hugging Face Space',
      details: error.message
    });
  }
});

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
  console.log(`🚀 Proxy server running on port ${PORT}`);
})
