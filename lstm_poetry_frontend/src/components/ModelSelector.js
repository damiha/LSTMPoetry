import React from 'react';

function ModelSelector({ model, setModel }) {
  return (
    <select 
      value={model} 
      onChange={(e) => setModel(e.target.value)}
      className="model-selector"
    >
      <option value="german_medium_structured">german_medium_structured</option>
      <option value="german_large_unstructured">german_large_unstructured</option>
    </select>
  );
}

export default ModelSelector;