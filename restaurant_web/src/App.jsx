import './App.css'
import Home from './Home';
import { Routes, Route } from "react-router-dom";
import ResultPage from './ResultPage';

function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/result" element={<ResultPage />} />
    </Routes>
  );
}

export default App
