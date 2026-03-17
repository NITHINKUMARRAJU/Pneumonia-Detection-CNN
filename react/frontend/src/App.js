// src/App.js

import React, { useState } from 'react';
import axios from 'axios';
import './App.css'; // You can add styling here

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  // Handles file selection
  const onFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setResult(null); // Reset previous result
    }
  };

  // Handles form submission to the backend
  const onFileUpload = async () => {
    if (!selectedFile) {
      alert("Please select a file first!");
      return;
    }
    
    setLoading(true);
    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      // The API call to your Python backend
      const response = await axios.post("http://127.0.0.1:5000/predict", formData);
      setResult(response.data); // Save the prediction result
    } catch (error) {
      console.error("Error uploading file:", error);
      alert("Error uploading file. Check the console for details.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>MedFusion Diagnostic Hub</h1>
        <p>Upload a Chest X-ray to get a prediction.</p>
        
        <input type="file" onChange={onFileChange} />
        
        {preview && <img src={preview} alt="Preview" style={{maxWidth: '300px', marginTop: '20px'}} />}
        
        <button onClick={onFileUpload} disabled={!selectedFile || loading} style={{marginTop: '20px'}}>
          {loading ? 'Analyzing...' : 'Get Prediction'}
        </button>

        {result && (
          <div style={{marginTop: '20px', border: '1px solid white', padding: '10px'}}>
            <h3>Analysis Result:</h3>
            <p><strong>Prediction:</strong> {result.prediction}</p>
            <p><strong>Confidence:</strong> {(result.confidence * 100).toFixed(2)}%</p>
          </div>
        )}
      </header>
    </div>
  );
}

export default App;