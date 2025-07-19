// server.js

const express = require('express');
const axios = require('axios');
const cors = require('cors');

const app = express();

// Middleware
app.use(cors()); // Ruhusu CORS kutoka domain yoyote
app.use(express.json()); // Soma JSON bodies

// Hugging Face Space API URL
const HF_API_URL = 'https://pcam-velox.hf.space/run/predict';

// Proxy endpoint to call Hugging Face Space
app.post('/predict', async (req, res) => {
  try {
    const response = await axios.post(HF_API_URL, req.body, {
      headers: {
        'Content-Type': 'application/json'
      }
    });

    // Send back response from Hugging Face to frontend
    res.json(response.data);

  } catch (error) {
    console.error('Error forwarding request to Hugging Face:', error.message);
    res.status(500).json({
      error: 'Failed to fetch prediction from Hugging Face Space',
      details: error.message
    });
  }
});

// Start the server
const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
  console.log(`🚀 Proxy server is running at http://localhost:${PORT}`);
})
