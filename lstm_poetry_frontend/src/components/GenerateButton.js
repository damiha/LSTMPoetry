import React from 'react';

function GenerateButton({ onClick }) {
  return (
    <button onClick={onClick} className="generate-button">
      Generate
    </button>
  );
}

export default GenerateButton;