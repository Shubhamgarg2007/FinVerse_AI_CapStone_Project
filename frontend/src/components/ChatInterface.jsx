// src/components/ChatInterface.jsx
import React, { useState } from 'react';
import axios from 'axios';

// FIX: A new helper function to format complex error messages
const formatErrorMessage = (error) => {
  // Default error message
  let message = 'An unknown error occurred. Please try again.';

  if (axios.isAxiosError(error) && error.response) {
    const errorDetail = error.response.data.detail;

    // Case 1: FastAPI validation error (usually an array of objects)
    if (Array.isArray(errorDetail)) {
      // Extract the first validation error message for simplicity
      const firstError = errorDetail[0];
      const field = firstError.loc ? firstError.loc.join(' -> ') : 'field';
      const msg = firstError.msg || 'is invalid';
      message = `Data validation error: The ${field} ${msg}.`;
    } 
    // Case 2: Custom string error from our backend
    else if (typeof errorDetail === 'string') {
      message = errorDetail;
    }
    // Case 3: A generic object, stringify it for debugging
    else if (typeof errorDetail === 'object' && errorDetail !== null) {
      message = JSON.stringify(errorDetail);
    }
  } else if (error.message) {
    // Case 4: Network error or other non-Axios errors
    message = error.message;
  }
  
  return message;
};


const ChatInterface = ({ user }) => {
  const [input, setInput] = useState('');
  const [chatLog, setChatLog] = useState([]);

  const handleInputChange = (e) => setInput(e.target.value);

  const handleSubmit = async (e) => {
    e.preventDefault();

    const userMessage = input.trim();
    if (!userMessage) return;

    const updatedChatLog = [...chatLog, { sender: 'user', message: userMessage }];
    setChatLog(updatedChatLog);
    setInput('');

    try {
      const payload = {
        age: user?.age,
        annual_income: user?.income,
        monthly_savings: user?.monthlySavings,
        risk_appetite: user?.riskAppetite,
        investment_goal: user?.primaryGoal,
        message: userMessage,
      };

      const response = await axios.post('http://localhost:8000/predict', payload, {
        headers: { 'Content-Type': 'application/json' },
      });

      const recommendation = `Debt: ${response.data.debt_allocation}%, Equity: ${response.data.equity_allocation}%, Mutual Funds: ${response.data.mutual_fund_allocation}%`;

      setChatLog([...updatedChatLog, { sender: 'ai', message: recommendation }]);
    } catch (error) {
      console.error('Full error:', error);
      // FIX: Use the new error formatting function
      const errorMessage = formatErrorMessage(error);
      setChatLog([...updatedChatLog, { sender: 'ai', message: `❌ ${errorMessage}` }]);
    }
  };

  // The rest of your return statement remains the same...
  return (
    <div className="chat-container">
      <div className="chat-log">
        {chatLog.map((entry, index) => (
          <div key={index} className={`chat-entry ${entry.sender}`}>
            {entry.message}
          </div>
        ))}
      </div>
      <div className="chat-prompts">
        <p className="prompt-heading">Try asking:</p>
        <ul>
          <li>What if my risk appetite was "High"?</li>
          <li>My monthly savings just increased to 30000.</li>
          <li>What if I wanted to focus on "Retirement"?</li>
          <li>I'm 25 with a high-risk tolerance, but I'm saving for a down payment on a house in 3 years.</li>
          <li>I'm 32, earn 20LPA, and can save 60k a month. I have no problem with high risk and want to retire early.</li>
        </ul>
      </div>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={input}
          onChange={handleInputChange}
          placeholder="Ask for a recommendation or provide an update..."
        />
        <button type="submit">Send</button>
      </form>
    </div>
  );
};

export default ChatInterface;