import { Routes, Route } from 'react-router-dom'
import { Toaster } from 'react-hot-toast'
import Navbar from './components/Navbar'
import HomePage from './pages/HomePage'
import ClassifyPage from './pages/ClassifyPage'
import DatasetPage from './pages/DatasetPage'
import PreprocessingPage from './pages/PreprocessingPage'
import PerformancePage from './pages/PerformancePage'
import AboutPage from './pages/AboutPage'

function App() {
  return (
    <div className="min-h-screen bg-slate-50">
      <Navbar />
      <main className="pt-16">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/classify" element={<ClassifyPage />} />
          <Route path="/dataset" element={<DatasetPage />} />
          <Route path="/preprocessing" element={<PreprocessingPage />} />
          <Route path="/performance" element={<PerformancePage />} />
          <Route path="/about" element={<AboutPage />} />
        </Routes>
      </main>
      <Toaster
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: '#1e293b',
            color: '#f1f5f9',
            borderRadius: '10px',
            fontSize: '14px',
          },
          success: {
            iconTheme: { primary: '#22c55e', secondary: '#fff' },
          },
          error: {
            iconTheme: { primary: '#ef4444', secondary: '#fff' },
          },
        }}
      />
    </div>
  )
}

export default App
