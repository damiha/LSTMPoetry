import React from 'react';

function InfoBox() {
  return (
    <div className="info-box">
      <p>
        The LSTM has 3 layers, hidden size = 512, embedding size = 32, sequence length = 128. 
        Two models are available: "german_medium_structured" which contains 660 kB of German 
        poetry downloaded from <a href="https://www.gutenberg.org/">Project Gutenberg</a>. 
        The large model, "german_large_unstructured", was trained on 4.3 MB of German poetry 
        webscraped from <a href="https://de.wikisource.org/wiki/Liste_der_Gedichte">Wikisource</a>.
      </p>
    </div>
  );
}

export default InfoBox;