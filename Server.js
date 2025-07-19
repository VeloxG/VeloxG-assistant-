const express = require('express');
const axios = require('axios');
const cors = require('cors');

const app = express();
app.use(cors());
app.use(express.json());

const HF_API_URL = 'https://pcam-velox.hf.space/run/predict';

app.post('/predict', async (req, res) => {
  try {
    const response = await axios.post(HF_API_URL, req.body, {
      headers: { 'Content-Type': 'application/json' }
    });
    res.json(response.data);
  } catch (error) {
    res.status(500).json({ error: 'Error forwarding request' });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Listening on port ${PORT}`))
