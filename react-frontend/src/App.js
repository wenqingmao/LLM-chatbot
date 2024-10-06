import React, { useState } from 'react';
import './App.css';  // Link to a CSS file

function App() {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');

  // Function to handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault(); // Prevent the page from refreshing

    try {
      const response = await fetch('http://localhost:5001/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: question }),
      });

      const data = await response.json();
      setAnswer(data.response); // Set the answer returned from the backend
    } catch (error) {
      console.error('Error fetching data:', error);
      setAnswer('Error fetching answer');
    }
  };

  return (
    <div className="app-container">
      <h1>Ask the Chatbot</h1>
      <form onSubmit={handleSubmit}>
        <label htmlFor="question">Enter your question:</label>
        <input
          type="text"
          id="question"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Type your question here..."
        />
        <button type="submit">Submit</button>
      </form>

      <div className="response">
        <h2>Answer:</h2>
        <p>{answer}</p>
      </div>
    </div>
  );
}

export default App;
