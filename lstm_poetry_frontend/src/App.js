import React, { useState } from 'react';
import './App.css';
import InfoBox from './components/InfoBox';
import ModelSelector from './components/ModelSelector';
import TextInput from './components/TextInput';
import GenerateButton from './components/GenerateButton';
import GeneratedTextView from './components/GeneratedTextView';

function App() {
  const [model, setModel] = useState('german_medium_structured');
  const [prompt, setPrompt] = useState('');
  const [numChars, setNumChars] = useState(100);
  const [generatedText, setGeneratedText] = useState('');

  const handleGenerate = async () => {
    const response = await fetch('http://127.0.0.1:5000/generate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ model, prompt, n_chars: numChars }),
    });

    const data = await response.json();
    setGeneratedText(prompt + data.generated);
  };

  return (
    <div className="App">
      <h1>LSTM Poetry</h1>
      <InfoBox />
      <ModelSelector model={model} setModel={setModel} />
      <div className="input-group">
        <TextInput 
          placeholder="Gedichtanfang..." 
          value={prompt} 
          onChange={(e) => setPrompt(e.target.value)} 
        />
        <TextInput 
          type="number" 
          value={numChars} 
          onChange={(e) => setNumChars(e.target.value)} 
        />
        <GenerateButton onClick={handleGenerate} />
      </div>
      <GeneratedTextView text={generatedText} />
    </div>
  );
}

export default App;