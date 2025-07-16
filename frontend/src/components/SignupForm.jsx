import React, { useState } from 'react';
import '../assets/styles/Auth.css';
import axios from 'axios';

const SignupForm = ({ onShowLogin, onSignupSuccess }) => {
  const [step, setStep] = useState(1);
  const [formData, setFormData] = useState({
    // Step 1
    fullName: '',
    email: '',
    password: '',
    // Step 2
    income: '',
    riskAppetite: 'Medium',
    primaryGoal: 'Retirement Planning',
    retirementAge: '65',
    monthlySavings: '',
    investmentExperience: 'Beginner',
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const nextStep = () => setStep((prev) => prev + 1);
  const prevStep = () => setStep((prev) => prev - 1);
  
const handleSubmit = async (e) => {
  e.preventDefault();
  console.log('--- Signup handleSubmit: Fired ---');

  // Create a new object with the correct data types
  const dataToSend = {
    ...formData,
    income: parseFloat(formData.income),
    retirementAge: parseInt(formData.retirementAge, 10),
    monthlySavings: parseFloat(formData.monthlySavings),
  };

  try {
    console.log('--- Signup handleSubmit: Inside try block, about to call API ---');
    console.log('--- Signup handleSubmit: Sending this data:', dataToSend); // Log the corrected data

    // Send the corrected data object
    const response = await axios.post('http://localhost:8000/api/signup', dataToSend);

    console.log('--- Signup handleSubmit: Signup API call successful ---', response.data);

    // ... rest of the function remains the same ...
    const loginData = new URLSearchParams();
    loginData.append('username', formData.email);
    loginData.append('password', formData.password);

    const tokenResponse = await axios.post('http://localhost:8000/api/token', loginData);

    console.log('--- Signup handleSubmit: Token API call successful ---', tokenResponse.data);

    localStorage.setItem('token', tokenResponse.data.access_token);
    onSignupSuccess();
  } catch (err) {
    console.error('--- Signup handleSubmit: CRASHED in catch block ---', err);
    // Check the backend terminal for a detailed 422 error
    if (err.response && err.response.status === 422) {
      alert('Validation Error: Please check the data you entered. Details in console.');
      console.error('FastAPI Validation Error:', err.response.data.detail);
    } else {
      alert(err.response ? err.response.data.detail : 'An error occurred during signup.');
    }
  }
};

  return (
    <div className="auth-form-container">
      <h2>Create Your Account</h2>
      <div className="step-indicator">Step {step} of 2</div>
      <form onSubmit={handleSubmit}>
        {step === 1 && (
          <>
            <h3>Personal Details</h3>
            <div className="form-group">
              <label htmlFor="fullName">Full Name</label>
              <input type="text" id="fullName" name="fullName" className="form-input" value={formData.fullName} onChange={handleChange} required />
            </div>
            <div className="form-group">
              <label htmlFor="email">Email</label>
              <input type="email" id="email" name="email" className="form-input" value={formData.email} onChange={handleChange} required />
            </div>
            <div className="form-group">
              <label htmlFor="password">Password</label>
              <input type="password" id="password" name="password" className="form-input" value={formData.password} onChange={handleChange} required minLength="8" />
            </div>
            <div className="form-navigation">
                <div></div> {/* Spacer */}
                <button type="button" className="btn-primary" onClick={nextStep} style={{width: 'auto', padding: '0.75rem 2rem'}}>Next</button>
            </div>
          </>
        )}

        {step === 2 && (
          <>
            <h3>Financial Profile</h3>
             <div className="form-group">
              <label htmlFor="income">Annual Income </label>
              <input type="number" id="income" name="income" className="form-input" placeholder="e.g., 75000" value={formData.income} onChange={handleChange} required />
            </div>
             <div className="form-group">
              <label htmlFor="monthlySavings">Monthly Savings</label>
              <input type="number" id="monthlySavings" name="monthlySavings" className="form-input" placeholder="e.g., 500" value={formData.monthlySavings} onChange={handleChange} required />
            </div>
            <div className="form-group">
              <label htmlFor="retirementAge">Target Retirement Age</label>
              <input type="number" id="retirementAge" name="retirementAge" className="form-input" placeholder="e.g., 65" value={formData.retirementAge} onChange={handleChange} required />
            </div>
            <div className="form-group">
              <label htmlFor="riskAppetite">Risk Appetite</label>
              <select id="riskAppetite" name="riskAppetite" className="form-select" value={formData.riskAppetite} onChange={handleChange}>
                <option value="Low">Low</option>
                <option value="Medium">Medium</option>
                <option value="High">High</option>
                <option value="Very High">Very High</option>
              </select>
            </div>
            <div className="form-group">
              <label htmlFor="investmentExperience">Investment Experience</label>
              <select id="investmentExperience" name="investmentExperience" className="form-select" value={formData.investmentExperience} onChange={handleChange}>
                <option value="Beginner">Beginner (Just starting out)</option>
                <option value="Intermediate">Intermediate (Some experience)</option>
                <option value="Advanced">Advanced (Very experienced)</option>
              </select>
            </div>
            <div className="form-group">
              <label htmlFor="primaryGoal">Primary Financial Goal</label>
              <select id="primaryGoal" name="primaryGoal" className="form-select" value={formData.primaryGoal} onChange={handleChange}>
                <option value="Retirement Planning">Retirement Planning</option>
                <option value="Wealth Creation">Wealth Creation</option>
                <option value="Major Purchase">Major Purchase (e.g., Home, Car)</option>
                <option value="Education Fund">Education Fund</option>
                <option value="Income Generation">Income Generation</option>
              </select>
            </div>
            <div className="form-navigation">
                <button type="button" className="btn-secondary" onClick={prevStep}>Back</button>
                <button type="submit" className="btn-primary">Create Account</button>
            </div>
          </>
        )}
      </form>
       <div className="auth-switch-text">
        Already have an account?{' '}
        <button onClick={onShowLogin}>Login</button>
      </div>
    </div>
  );
};

export default SignupForm;