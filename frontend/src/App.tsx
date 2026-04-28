import { BrowserRouter as Router, Routes, Route, NavLink } from 'react-router-dom';
import UploadPage from './pages/Upload';
import AnalyticsPage from './pages/Analytics';
import { Activity, BarChart2 } from 'lucide-react';

const navClass = ({ isActive }: { isActive: boolean }) =>
  `text-sm font-medium flex items-center gap-2 px-3 py-1.5 rounded-lg transition-colors ${
    isActive
      ? 'text-[var(--accent)] bg-[var(--accent)]/10'
      : 'text-gray-400 hover:text-[var(--accent)] hover:bg-gray-800'
  }`;

function App() {
  return (
    <Router>
      <div className="min-h-screen flex flex-col">
        {/* Navbar */}
        <header className="bg-[var(--bg-card)] border-b border-gray-800 px-4 py-3">
          <div className="container mx-auto flex items-center justify-between">
            <div className="flex items-center gap-2 text-[var(--accent)]">
              <Activity size={24} />
              <h1 className="text-lg font-bold tracking-wide">Bee Monitor</h1>
            </div>
            <nav className="flex gap-2">
              <NavLink to="/" end className={navClass}>
                <Activity size={16} /> <span>Upload & Process</span>
              </NavLink>
              <NavLink to="/analytics" className={navClass}>
                <BarChart2 size={16} /> <span>Analytics & Research</span>
              </NavLink>
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
