import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import UploadPage from './pages/Upload';
import AnalyticsPage from './pages/Analytics';
import { Activity, BarChart2 } from 'lucide-react';

function App() {
  return (
    <Router>
      <div className="min-h-screen flex flex-col">
        {/* Navbar */}
        <header className="bg-[var(--bg-card)] border-b border-gray-800 p-4">
          <div className="container mx-auto flex items-center justify-between">
            <div className="flex items-center space-x-2 text-[var(--accent)] cursor-pointer">
              <Activity size={28} />
              <h1 className="text-xl font-bold tracking-wide">Bee Monitor</h1>
            </div>
            <nav className="flex space-x-6">
              <Link to="/" className="text-sm text-gray-300 hover:text-[var(--accent)] font-medium flex items-center space-x-2 transition-colors">
                <Activity size={18} /> <span>Upload & Process</span>
              </Link>
              <Link to="/analytics" className="text-sm text-gray-300 hover:text-[var(--accent)] font-medium flex items-center space-x-2 transition-colors">
                <BarChart2 size={18} /> <span>Analytics & Research</span>
              </Link>
            </nav>
          </div>
        </header>

        {/* Main Content */}
        <main className="flex-grow container mx-auto p-4 md:p-6 lg:p-8 flex flex-col">
          <Routes>
            <Route path="/" element={<UploadPage />} />
            <Route path="/analytics" element={<AnalyticsPage />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
