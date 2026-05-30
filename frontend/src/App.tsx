import { BrowserRouter as Router, Routes, Route, NavLink } from 'react-router-dom';
import UploadPage from './pages/Upload';
import AnalyticsPage from './pages/Analytics';
import EvaluationPage from './pages/Evaluation';
import { Activity, BarChart2, Target } from 'lucide-react';

const navClass = ({ isActive }: { isActive: boolean }) =>
  `text-sm font-medium flex items-center gap-2 px-3 py-1.5 rounded-lg transition-colors ${
    isActive
      ? 'text-[var(--accent)] bg-[var(--accent)]/10'
      : 'text-gray-400 hover:text-[var(--accent)] hover:bg-gray-800'
  }`;

function App() {
  return (
    <Router>
      <div className="min-h-screen flex flex-col relative overflow-hidden">
        {/* Background Decorative Elements */}
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-indigo-600/20 rounded-full blur-[120px] pointer-events-none"></div>
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-amber-600/20 rounded-full blur-[120px] pointer-events-none"></div>
        
        {/* Navbar */}
        <header className="sticky top-0 z-50 glass-panel border-x-0 border-t-0 rounded-none px-4 py-4 mb-4 shadow-lg shadow-black/20">
          <div className="container mx-auto flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="bg-gradient-to-br from-amber-400 to-orange-500 p-2 rounded-xl shadow-lg shadow-amber-500/30">
                <Activity size={24} className="text-white" />
              </div>
              <h1 className="text-2xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white to-gray-400">Bee Monitor</h1>
            </div>
            <nav className="flex gap-2 glass-panel px-2 py-1.5 rounded-xl">
              <NavLink to="/" end className={navClass}>
                <Activity size={18} /> <span>Завантаження</span>
              </NavLink>
              <NavLink to="/analytics" className={navClass}>
                <BarChart2 size={18} /> <span>Аналітика</span>
              </NavLink>
              <NavLink to="/evaluation" className={navClass}>
                <Target size={18} /> <span>Точність</span>
              </NavLink>
            </nav>
          </div>
        </header>

        {/* Main Content */}
        <main className="flex-grow container mx-auto p-4 md:p-6 lg:p-8 flex flex-col relative z-10 animate-fade-in">
          <Routes>
            <Route path="/" element={<UploadPage />} />
            <Route path="/analytics" element={<AnalyticsPage />} />
            <Route path="/evaluation" element={<EvaluationPage />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
