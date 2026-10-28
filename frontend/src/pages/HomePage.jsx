import { Link } from 'react-router-dom'
import { ArrowRight, Brain, Database, Cpu, BarChart2, CheckCircle, Layers } from 'lucide-react'

const CATEGORIES = [
  { name: 'World', emoji: '🌍', color: 'blue', desc: 'Politics, diplomacy, international affairs', bg: 'bg-blue-50', border: 'border-blue-200', text: 'text-blue-700', badge: 'bg-blue-100 text-blue-700' },
  { name: 'Sports', emoji: '⚽', color: 'green', desc: 'Football, cricket, athletics, championships', bg: 'bg-green-50', border: 'border-green-200', text: 'text-green-700', badge: 'bg-green-100 text-green-700' },
  { name: 'Business', emoji: '💼', color: 'orange', desc: 'Finance, markets, companies, economy', bg: 'bg-orange-50', border: 'border-orange-200', text: 'text-orange-700', badge: 'bg-orange-100 text-orange-700' },
  { name: 'Sci/Tech', emoji: '💻', color: 'purple', desc: 'AI, research, gadgets, innovation', bg: 'bg-purple-50', border: 'border-purple-200', text: 'text-purple-700', badge: 'bg-purple-100 text-purple-700' },
]

const PIPELINE_STEPS = [
  { icon: <Database className="w-5 h-5" />, label: 'Dataset', sub: 'AG News 120K+', color: 'bg-blue-500' },
  { icon: <Cpu className="w-5 h-5" />, label: 'Preprocessing', sub: 'Clean & Normalize', color: 'bg-indigo-500' },
  { icon: <Layers className="w-5 h-5" />, label: 'Features', sub: 'TF-IDF / BERT', color: 'bg-violet-500' },
  { icon: <Brain className="w-5 h-5" />, label: 'Model', sub: 'LR / DistilBERT', color: 'bg-purple-500' },
  { icon: <CheckCircle className="w-5 h-5" />, label: 'Prediction', sub: '4 Categories', color: 'bg-fuchsia-500' },
]

const HOW_IT_WORKS = [
  { step: '01', title: 'Input News', desc: 'Paste a news headline and article body into the classifier interface.' },
  { step: '02', title: 'Preprocessing', desc: 'Text is cleaned: HTML tags removed, lowercased, tokenized, stopwords filtered, and lemmatized.' },
  { step: '03', title: 'Feature Extraction', desc: 'TF-IDF vectorizer or DistilBERT tokenizer converts text to numerical features.' },
  { step: '04', title: 'Classification', desc: 'The trained model predicts the category and returns confidence scores for all 4 classes.' },
]

const TECH_BADGES = [
  { label: 'Python 3.10+', color: 'bg-yellow-100 text-yellow-800 border-yellow-200' },
  { label: 'scikit-learn', color: 'bg-blue-100 text-blue-800 border-blue-200' },
  { label: 'HuggingFace Transformers', color: 'bg-orange-100 text-orange-800 border-orange-200' },
  { label: 'DistilBERT', color: 'bg-purple-100 text-purple-800 border-purple-200' },
  { label: 'FastAPI', color: 'bg-green-100 text-green-800 border-green-200' },
  { label: 'React 18', color: 'bg-cyan-100 text-cyan-800 border-cyan-200' },
  { label: 'Tailwind CSS', color: 'bg-teal-100 text-teal-800 border-teal-200' },
  { label: 'AG News Dataset', color: 'bg-red-100 text-red-800 border-red-200' },
]

export default function HomePage() {
  return (
    <div className="min-h-screen">
      {/* Hero */}
      <section className="relative overflow-hidden bg-gradient-to-br from-slate-900 via-indigo-950 to-slate-900 text-white">
        <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxnIGZpbGw9IiMyMjMzNjYiIGZpbGwtb3BhY2l0eT0iMC4xIj48cGF0aCBkPSJNMzYgMzBoLTZWMThoNnYxMnptMCAwaC02VjQyaDZ2LTEyeiIvPjwvZz48L2c+PC9zdmc+')] opacity-20" />
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24 md:py-32">
          <div className="text-center max-w-4xl mx-auto">
            <div className="inline-flex items-center gap-2 bg-indigo-900/60 border border-indigo-700/50 rounded-full px-4 py-1.5 text-indigo-300 text-sm font-medium mb-6">
              <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
              CS459 Natural Language Processing — SVNIT Surat
            </div>
            <h1 className="text-4xl sm:text-5xl md:text-6xl font-extrabold tracking-tight mb-6 leading-tight">
              Automatic News Topic
              <span className="block bg-gradient-to-r from-indigo-400 via-violet-400 to-purple-400 bg-clip-text text-transparent">
                Classification
              </span>
            </h1>
            <p className="text-lg sm:text-xl text-slate-300 max-w-2xl mx-auto mb-10 leading-relaxed">
              Classify news articles into World, Sports, Business, or Sci/Tech using TF-IDF + Logistic Regression and DistilBERT transformer models trained on the AG News dataset.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link
                to="/classify"
                className="inline-flex items-center gap-2 bg-indigo-600 hover:bg-indigo-500 text-white font-semibold px-8 py-3.5 rounded-xl transition-all duration-200 shadow-lg shadow-indigo-900/50 hover:shadow-indigo-700/50 group"
              >
                Classify News
                <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
              </Link>
              <Link
                to="/performance"
                className="inline-flex items-center gap-2 bg-white/10 hover:bg-white/20 border border-white/20 text-white font-semibold px-8 py-3.5 rounded-xl transition-all duration-200 backdrop-blur-sm"
              >
                <BarChart2 className="w-4 h-4" />
                View Performance
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Quick stats bar */}
      <div className="bg-indigo-600 text-white">
        <div className="max-w-7xl mx-auto px-4 py-3 flex flex-wrap justify-center gap-6 text-sm font-medium">
          {[
            { label: '4 Categories', icon: '🏷️' },
            { label: '120K+ Articles', icon: '📰' },
            { label: 'TF-IDF + DistilBERT', icon: '🤖' },
            { label: 'AG News Dataset', icon: '🗄️' },
          ].map(({ label, icon }) => (
            <div key={label} className="flex items-center gap-1.5">
              <span>{icon}</span>
              <span>{label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* NLP Pipeline */}
      <section className="py-16 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-slate-900 mb-3">NLP Pipeline Overview</h2>
            <p className="text-slate-500 max-w-xl mx-auto">End-to-end pipeline from raw text to classified categories</p>
          </div>
          <div className="flex flex-wrap items-center justify-center gap-2">
            {PIPELINE_STEPS.map((step, i) => (
              <div key={step.label} className="flex items-center gap-2">
                <div className="flex flex-col items-center gap-2">
                  <div className={`w-14 h-14 ${step.color} rounded-xl flex items-center justify-center text-white shadow-lg`}>
                    {step.icon}
                  </div>
                  <div className="text-center">
                    <p className="text-sm font-bold text-slate-800">{step.label}</p>
                    <p className="text-xs text-slate-500">{step.sub}</p>
                  </div>
                </div>
                {i < PIPELINE_STEPS.length - 1 && (
                  <ArrowRight className="w-5 h-5 text-slate-300 mb-5 flex-shrink-0" />
                )}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Categories */}
      <section className="py-16 bg-slate-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-slate-900 mb-3">Supported Categories</h2>
            <p className="text-slate-500 max-w-xl mx-auto">4 news topic categories from the AG News corpus</p>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {CATEGORIES.map((cat) => (
              <div
                key={cat.name}
                className={`${cat.bg} ${cat.border} border rounded-xl p-6 hover:shadow-md transition-shadow`}
              >
                <div className="text-4xl mb-3">{cat.emoji}</div>
                <h3 className={`text-xl font-bold ${cat.text} mb-2`}>{cat.name}</h3>
                <p className="text-sm text-slate-600 mb-3">{cat.desc}</p>
                <span className={`badge ${cat.badge}`}>AG News Class</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* How it works */}
      <section className="py-16 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-slate-900 mb-3">How It Works</h2>
            <p className="text-slate-500 max-w-xl mx-auto">Four steps from raw news text to a confident prediction</p>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {HOW_IT_WORKS.map((step) => (
              <div key={step.step} className="relative card hover:shadow-md transition-shadow">
                <div className="text-4xl font-black text-indigo-100 absolute top-4 right-4">
                  {step.step}
                </div>
                <div className="relative">
                  <h3 className="text-lg font-bold text-slate-900 mb-2">{step.title}</h3>
                  <p className="text-sm text-slate-500 leading-relaxed">{step.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Tech Stack */}
      <section className="py-16 bg-slate-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-10">
            <h2 className="text-3xl font-bold text-slate-900 mb-3">Technology Stack</h2>
            <p className="text-slate-500">Built with state-of-the-art NLP and web technologies</p>
          </div>
          <div className="flex flex-wrap justify-center gap-3">
            {TECH_BADGES.map((tech) => (
              <span
                key={tech.label}
                className={`px-4 py-2 rounded-full border text-sm font-semibold ${tech.color}`}
              >
                {tech.label}
              </span>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-16 bg-gradient-to-br from-indigo-600 to-violet-700 text-white text-center">
        <div className="max-w-2xl mx-auto px-4">
          <h2 className="text-3xl font-bold mb-4">Ready to classify?</h2>
          <p className="text-indigo-200 mb-8">Try our live classifier with TF-IDF or DistilBERT models</p>
          <Link
            to="/classify"
            className="inline-flex items-center gap-2 bg-white text-indigo-700 font-bold px-8 py-3.5 rounded-xl hover:bg-indigo-50 transition-colors shadow-lg group"
          >
            Start Classifying
            <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
          </Link>
        </div>
      </section>
    </div>
  )
}
