import React from 'react';

function GeneratedTextView({ text }) {
    // Split the text into lines and render them with <br /> tags in between
    const formattedText = text.split('\n').map((line, index) => (
        <React.Fragment key={index}>
            {line}
            <br />
        </React.Fragment>
    ));

    return (
        <div className="generated-text-view">
            {formattedText}
        </div>
    );
}

export default GeneratedTextView;