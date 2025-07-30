import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import '../assets/styles/Dashboard.css';
import '../assets/styles/ChatInterface.css';

// A new helper component to render chat messages with line breaks
// This makes the AI report look formatted
const ChatMessage = ({ sender, message }) => {
  // Use a regex to find emojis at the start of lines and style them
  const formatLine = (line) => {
    let content = line;
    let emoji = '';

    // Check for common emojis at the start and extract them
    const emojiMatch = line.match(/^([📊📈💰⏱️👤🎯🤖💸💳🏛️🏦🏆💡⚠️📋🔄📚🔍🎢💎🔮✅❌👋⚡🟢🟡🔴🚀🧠📱📝🛡️🥇🔬⚖️])/);
    if (emojiMatch) {
      emoji = emojiMatch[1];
      content = line.substring(emoji.length).trim();
    }

    // Apply specific styles for different line types if needed
    if (line.startsWith('=')) {
      return <p className="section-divider">{line}</p>;
    }
    if (line.startsWith('---')) {
      return <p className="subsection-divider">{line}</p>;
    }
    if (line.startsWith('•')) {
      return <p className="list-item"><span className="bullet-point">•</span> {content}</p>;
    }
    if (line.startsWith('1.') || line.startsWith('2.') || line.startsWith('3.') || line.startsWith('4.') || line.startsWith('5.') || line.startsWith('6.') || line.startsWith('7.')) {
        return <p className="ordered-list-item"><span className="list-number">{line.split('.')[0]}.</span> {line.substring(line.indexOf('.') + 1).trim()}</p>;
    }


    return <p>{emoji && <span className="line-emoji">{emoji} </span>}{content}</p>;
  };

  return (
    <div className={`message ${sender}`}>
      {message.split('\n').map((line, i) => (
        <React.Fragment key={i}>
          {formatLine(line)}
        </React.Fragment>
      ))}
    </div>
  );
};


const ChatInterface = ({ user }) => {
  const [inputMessage, setInputMessage] = useState('');
  const [chatLog, setChatLog] = useState([]);
  const chatLogRef = useRef(null);

  // State for user profile data, initialized from props if available
  const [userData, setUserData] = useState({
    age: user?.age || '',
    annual_income: user?.income || '',
    monthly_savings: user?.monthlySavings || '',
    risk_appetite: user?.riskAppetite || 'Medium',
    investment_goal: user?.primaryGoal || 'Wealth Creation',
    timeline_months: '', // This might not be in user prop, keep it as it's parsed from chat
    emergency_fund: '',
    existing_investment_pct: '',
    goal_amount: ''
  });

  // Effect to load chat history from localStorage on component mount
  useEffect(() => {
  if (user) {
    const userKey = user.id || 'default_user';
    const savedChat = localStorage.getItem(`chatHistory_${userKey}`);
    if (savedChat) {
      setChatLog(JSON.parse(savedChat));
    }
  }
}, [user]);

useEffect(() => {
  if (user) {
    const userKey = user.id || 'default_user';
    localStorage.setItem(`chatHistory_${userKey}`, JSON.stringify(chatLog));
  }
}, [chatLog, user]);

  // Effect to update internal user data state if the 'user' prop changes 
  useEffect(() => {
    if (user) {
      setUserData(prevData => ({
        ...prevData,
        age: user.age || '',
        annual_income: user.income || '',
        monthly_savings: user.monthlySavings || '',
        risk_appetite: user.riskAppetite || 'Medium',
        investment_goal: user.primaryGoal || 'Wealth Creation',
        // timeline_months, emergency_fund, existing_investment_pct, goal_amount
        // These are typically derived from chat, not directly from static user profile in DB
      }));
    }
  }, [user]);


  // Scroll to the bottom of the chat log whenever it updates
  useEffect(() => {
    if (chatLogRef.current) {
      chatLogRef.current.scrollTop = chatLogRef.current.scrollHeight;
    }
  }, [chatLog]);

  const handleInputChange = (e) => setInputMessage(e.target.value);

  const handleProfileInputChange = (e) => {
    const { name, value } = e.target;
    setUserData(prevData => ({
      ...prevData,
      [name]: value
    }));
  };

  const handlePromptClick = (prompt) => {
    setInputMessage(prompt);
  };


  const handleSubmit = async (e) => {
    e.preventDefault();

    const userMessage = inputMessage.trim();
    // Allow sending just profile updates without a text message
    if (!userMessage && Object.values(userData).every(val => val === '' || val === 'Medium' || val === 'Wealth Creation')) {
      return; // Prevent sending empty messages and empty profiles
    }

    // Add user message to chat log
    const currentChatLog = [...chatLog, { sender: 'user', message: userMessage || 'Profile Update Submitted' }];
    setChatLog(currentChatLog);
    setInputMessage(''); // Clear the input field

    try {
      const payload = {
        message: userMessage,
        // Ensure numbers are sent as numbers, not empty strings
        age: userData.age ? parseInt(userData.age) : undefined,
        annual_income: userData.annual_income ? parseFloat(userData.annual_income) : undefined,
        monthly_savings: userData.monthly_savings ? parseFloat(userData.monthly_savings) : undefined,
        risk_appetite: userData.risk_appetite,
        investment_goal: userData.investment_goal,
        timeline_months: userData.timeline_months ? parseInt(userData.timeline_months) : undefined,
        emergency_fund: userData.emergency_fund ? parseFloat(userData.emergency_fund) : undefined,
        existing_investment_pct: userData.existing_investment_pct ? parseFloat(userData.existing_investment_pct) : undefined,
        goal_amount: userData.goal_amount ? parseFloat(userData.goal_amount) : undefined
      };

      const response = await axios.post('http://localhost:8000/predict', payload, {
        headers: { 'Content-Type': 'application/json' },
      });

      let aiResponseContent = '';
      if (response.data.full_report) {
        aiResponseContent = response.data.full_report;
      } else {
        // Fallback for unexpected or incomplete responses
        aiResponseContent = `Debt: ${response.data.debt_allocation}%, Equity: ${response.data.equity_allocation}%, Mutual Funds: ${response.data.mutual_fund_allocation}%`;
      }

      setChatLog(prevChatLog => [...prevChatLog, { sender: 'ai', message: aiResponseContent }]);

    } catch (error) {
      console.error('Failed to get AI response:', error);
      // Use the helper function to format the error message
      const errorMessage = formatErrorMessage(error);
      setChatLog(prevChatLog => [...prevChatLog, { sender: 'ai', message: `❌ ${errorMessage}` }]);
    }
  };

  return (
    <div className="chat-interface-wrapper">
      <div className="profile-input-section">
        <h3>Your Financial Profile</h3>
        <div className="input-group">
          <label htmlFor="age">Age:</label>
          <input type="number" id="age" name="age" value={userData.age} onChange={handleProfileInputChange} placeholder="e.g., 30" />
        </div>
        <div className="input-group">
          <label htmlFor="annual_income">Annual Income (₹):</label>
          <input type="number" id="annual_income" name="annual_income" value={userData.annual_income} onChange={handleProfileInputChange} placeholder="e.g., 1000000" />
        </div>
        <div className="input-group">
          <label htmlFor="monthly_savings">Monthly Savings (₹):</label>
          <input type="number" id="monthly_savings" name="monthly_savings" value={userData.monthly_savings} onChange={handleProfileInputChange} placeholder="e.g., 20000" />
        </div>
        <div className="input-group">
          <label htmlFor="risk_appetite">Risk Appetite:</label>
          <select id="risk_appetite" name="risk_appetite" value={userData.risk_appetite} onChange={handleProfileInputChange}>
            <option value="Low">Low</option>
            <option value="Medium">Medium</option>
            <option value="High">High</option>
          </select>
        </div>
        <div className="input-group">
          <label htmlFor="investment_goal">Primary Goal:</label>
          <input type="text" id="investment_goal" name="investment_goal" value={userData.investment_goal} onChange={handleProfileInputChange} placeholder="e.g., Retirement" />
        </div>
        <div className="input-group">
          <label htmlFor="timeline_months">Goal Timeline (months):</label>
          <input type="number" id="timeline_months" name="timeline_months" value={userData.timeline_months} onChange={handleProfileInputChange} placeholder="e.g., 60 (5 years)" />
        </div>
        <div className="input-group">
          <label htmlFor="emergency_fund">Emergency Fund (₹):</label>
          <input type="number" id="emergency_fund" name="emergency_fund" value={userData.emergency_fund} onChange={handleProfileInputChange} placeholder="e.g., 100000" />
        </div>
        <div className="input-group">
          <label htmlFor="existing_investment_pct">Existing Investment (%):</label>
          <input type="number" id="existing_investment_pct" name="existing_investment_pct" step="0.01" value={userData.existing_investment_pct} onChange={handleProfileInputChange} placeholder="e.g., 0.1 (10%)" />
        </div>
        <div className="input-group">
          <label htmlFor="goal_amount">Target Goal Amount (₹):</label>
          <input type="number" id="goal_amount" name="goal_amount" value={userData.goal_amount} onChange={handleProfileInputChange} placeholder="e.g., 500000" />
        </div>
      </div>

      <div className="chat-area">
        {/* MOVED: "Try asking" section is now here, at the top of chat-area */}
        <div className="chat-prompts">
          <p className="prompt-heading">Try asking:</p>
          <ul>
            <li onClick={() => handlePromptClick("What if my risk appetite was 'High'?")}>What if my risk appetite was "High"?</li>
            <li onClick={() => handlePromptClick("My monthly savings just increased to 30000.")}>My monthly savings just increased to 30000.</li>
            <li onClick={() => handlePromptClick("What if I wanted to focus on 'Retirement'?")}>What if I wanted to focus on "Retirement"?</li>
            <li onClick={() => handlePromptClick("I'm 25 with a high-risk tolerance, but I'm saving for a down payment on a house in 3 years.")}>I'm 25 with a high-risk tolerance, but I'm saving for a down payment on a house in 3 years.</li>
            <li onClick={() => handlePromptClick("I'm 32, earn 20LPA, and can save 60k a month. I have no problem with high risk and want to retire early.")}>I'm 32, earn 20LPA, and can save 60k a month. I have no problem with high risk and want to retire early.</li>
             <li onClick={() => handlePromptClick("Show me AI predictions for the market.")}>Show me AI predictions for the market.</li>
             <li onClick={() => handlePromptClick("How does your AI prediction work?")}>How does your AI prediction work?</li>
             <li onClick={() => handlePromptClick("What about AI accuracy?")}>What about AI accuracy?</li>
             <li onClick={() => handlePromptClick("Enable AI")}>Enable AI</li>
          </ul>
        </div>

        {/* MOVED: Chat log is now below the prompts */}
        <div className="chat-log" ref={chatLogRef}>
          {chatLog.map((entry, index) => (
            <ChatMessage key={index} sender={entry.sender} message={entry.message} />
          ))}
        </div>

        {/* The input form remains at the bottom */}
        <form onSubmit={handleSubmit} className="message-input-form">
          <input
            type="text"
            value={inputMessage}
            onChange={handleInputChange}
            placeholder="Ask for a recommendation or provide an update..."
          />
          <button type="submit">Send</button>
        </form>
      </div>
    </div>
  );
};

export default ChatInterface;

// Helper function (should be defined somewhere accessible, e.g., in a utils file or above ChatInterface)
const formatErrorMessage = (error) => {
    let message = 'An unknown error occurred. Please try again.';
    if (axios.isAxiosError(error) && error.response) {
        const errorDetail = error.response.data.detail;
        if (Array.isArray(errorDetail)) {
            const firstError = errorDetail[0];
            const field = firstError.loc ? firstError.loc.join(' -> ') : 'field';
            const msg = firstError.msg || 'is invalid';
            message = `Data validation error: The ${field} ${msg}.`;
        } else if (typeof errorDetail === 'string') {
            message = errorDetail;
        } else if (typeof errorDetail === 'object' && errorDetail !== null) {
            message = JSON.stringify(errorDetail);
        }
    } else if (error.message) {
        message = error.message;
    }
    return message;
};