import { Github, BookOpen, Database, Cpu, FlaskConical, Globe, Layers, ChevronRight } from 'lucide-react'

const TECH_STACK = [
  { label: 'React 18', color: 'bg-blue-100 text-blue-800', cat: 'Frontend' },
  { label: 'Vite', color: 'bg-yellow-100 text-yellow-800', cat: 'Frontend' },
  { label: 'Tailwind CSS', color: 'bg-cyan-100 text-cyan-800', cat: 'Frontend' },
  { label: 'Recharts', color: 'bg-indigo-100 text-indigo-800', cat: 'Frontend' },
  { label: 'FastAPI', color: 'bg-green-100 text-green-800', cat: 'Backend' },
  { label: 'Python 3.12', color: 'bg-blue-100 text-blue-800', cat: 'Backend' },
  { label: 'Uvicorn', color: 'bg-slate-100 text-slate-800', cat: 'Backend' },
  { label: 'Pydantic', color: 'bg-red-100 text-red-800', cat: 'Backend' },
  { label: 'scikit-learn', color: 'bg-orange-100 text-orange-800', cat: 'ML' },
  { label: 'PyTorch', color: 'bg-red-100 text-red-800', cat: 'ML' },
  { label: 'HuggingFace Transformers', color: 'bg-yellow-100 text-yellow-800', cat: 'ML' },
  { label: 'DistilBERT', color: 'bg-purple-100 text-purple-800', cat: 'ML' },
  { label: 'NLTK', color: 'bg-green-100 text-green-800', cat: 'NLP' },
  { label: 'TF-IDF', color: 'bg-teal-100 text-teal-800', cat: 'NLP' },
  { label: 'AG News', color: 'bg-slate-100 text-slate-800', cat: 'Dataset' },
  { label: 'HuggingFace Datasets', color: 'bg-yellow-100 text-yellow-800', cat: 'Dataset' },
]

const FUTURE_WORK = [
  'Multilingual news classification (Hindi, Spanish, French)',
  'Real-time news feed integration via RSS/APIs',
  'News summarization with abstractive models',
  'Fake news detection module',
  'Named Entity Recognition overlay',
  'Topic modeling (LDA) for unsupervised discovery',
  'Cloud deployment (AWS/GCP/Azure)',
  'Mobile application (React Native)',
  'Continuous model improvement with active learning',
  'Explainability with SHAP values',
]

const PIPELINE_STEPS = [
  { step: '1', label: 'AG News Dataset', sub: '127,600 articles · 4 classes', color: 'bg-blue-500' },
  { step: '2', label: 'Exploration', sub: 'Stats, word clouds, distributions', color: 'bg-teal-500' },
  { step: '3', label: 'Preprocessing', sub: 'TF-IDF & Transformer pipelines', color: 'bg-indigo-500' },
  { step: '4', label: 'Feature Extraction', sub: 'TF-IDF vectors / BERT tokens', color: 'bg-purple-500' },
  { step: '5', label: 'Model Training', sub: 'Logistic Regression + DistilBERT', color: 'bg-pink-500' },
  { step: '6', label: 'Evaluation', sub: 'F1, accuracy, confusion matrix', color: 'bg-orange-500' },
  { step: '7', label: 'Post-processing', sub: 'Confidence, label mapping', color: 'bg-amber-500' },
  { step: '8', label: 'Web Application', sub: 'FastAPI + React interactive UI', color: 'bg-green-500' },
]

export default function AboutPage() {
  return (
    <div className="max-w-6xl mx-auto px-4 py-10">
      {/* Hero */}
      <div className="bg-gradient-to-r from-indigo-600 to-purple-600 rounded-2xl p-8 text-white mb-10">
        <div className="flex items-start gap-4">
          <div className="text-4xl">📰</div>
          <div>
            <h1 className="text-3xl font-bold mb-2">Automatic News Topic Classification</h1>
            <p className="text-indigo-100 text-lg mb-4">
              An end-to-end NLP system that automatically identifies the topic of a news article
              using traditional ML and transformer-based deep learning approaches.
            </p>
            <div className="flex flex-wrap gap-3 text-sm">
              <span className="bg-white/20 px-3 py-1 rounded-full">CS459 — Natural Language Processing</span>
              <span className="bg-white/20 px-3 py-1 rounded-full">SVNIT Surat</span>
              <span className="bg-white/20 px-3 py-1 rounded-full">B.Tech IV CSE</span>
              <span className="bg-white/20 px-3 py-1 rounded-full">Academic Year 2025-26</span>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-10">
        {/* Problem Statement */}
        <div className="lg:col-span-2 space-y-6">
          <div className="bg-white rounded-xl shadow-sm p-6">
            <h2 className="text-xl font-bold text-slate-800 mb-3 flex items-center gap-2">
              <BookOpen className="w-5 h-5 text-indigo-500" />
              Problem Statement
            </h2>
            <p className="text-slate-600 leading-relaxed mb-3">
              The digital age produces millions of news articles daily across thousands of sources.
              Manually categorizing these articles into topics is infeasible at scale.
              Automated classification using Natural Language Processing enables news aggregators,
              recommendation systems, and media monitoring platforms to organize content efficiently.
            </p>
            <p className="text-slate-600 leading-relaxed">
              Given a news article (headline and/or body text), the system predicts which of the four
              AG News categories it belongs to: <b>World</b>, <b>Sports</b>, <b>Business</b>, or <b>Sci/Tech</b>.
            </p>
          </div>

          <div className="bg-white rounded-xl shadow-sm p-6">
            <h2 className="text-xl font-bold text-slate-800 mb-3 flex items-center gap-2">
              <Layers className="w-5 h-5 text-indigo-500" />
              Project Objectives
            </h2>
            <ul className="space-y-2">
              {[
                'Design and implement a complete NLP preprocessing pipeline',
                'Train and compare TF-IDF + LR (baseline) vs DistilBERT (advanced)',
                'Develop a FastAPI REST backend serving real-time predictions',
                'Build an interactive React web application for demonstration',
                'Perform comprehensive quantitative and qualitative evaluation',
                'Document the entire NLP pipeline for academic reporting',
              ].map((obj, i) => (
                <li key={i} className="flex items-start gap-2 text-sm text-slate-600">
                  <ChevronRight className="w-4 h-4 text-indigo-500 mt-0.5 flex-shrink-0" />
                  {obj}
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Team / Course Info */}
        <div className="space-y-4">
          <div className="bg-white rounded-xl shadow-sm p-6">
            <h2 className="text-lg font-bold text-slate-800 mb-4">Team</h2>
            <div className="space-y-3 text-sm">
              <div><span className="text-slate-500">Student:</span> <span className="font-semibold text-slate-800">[Your Name]</span></div>
              <div><span className="text-slate-500">Roll No:</span> <span className="font-semibold text-slate-800">[Roll Number]</span></div>
              <div><span className="text-slate-500">Course:</span> <span className="font-semibold text-slate-800">CS459 — NLP</span></div>
              <div><span className="text-slate-500">Institution:</span> <span className="font-semibold text-slate-800">SVNIT Surat</span></div>
              <div><span className="text-slate-500">Academic Year:</span> <span className="font-semibold text-slate-800">2025-26</span></div>
              <div><span className="text-slate-500">Supervisor:</span> <span className="font-semibold text-slate-800">[Professor Name]</span></div>
            </div>
          </div>

          <div className="bg-white rounded-xl shadow-sm p-6">
            <h2 className="text-lg font-bold text-slate-800 mb-4 flex items-center gap-2">
              <Database className="w-5 h-5 text-blue-500" />
              Dataset
            </h2>
            <div className="space-y-2 text-sm">
              <div className="font-semibold text-slate-800">AG News Corpus</div>
              <div className="text-slate-500">Zhang et al., 2015 (NeurIPS)</div>
              <div className="flex justify-between"><span className="text-slate-500">Train:</span><span className="font-medium">120,000</span></div>
              <div className="flex justify-between"><span className="text-slate-500">Test:</span><span className="font-medium">7,600</span></div>
              <div className="flex justify-between"><span className="text-slate-500">Classes:</span><span className="font-medium">4 (balanced)</span></div>
              <div className="flex justify-between"><span className="text-slate-500">Source:</span><span className="font-medium text-indigo-600">HuggingFace</span></div>
            </div>
          </div>
        </div>
      </div>

      {/* NLP Pipeline */}
      <div className="bg-white rounded-xl shadow-sm p-6 mb-8">
        <h2 className="text-xl font-bold text-slate-800 mb-6 flex items-center gap-2">
          <Cpu className="w-5 h-5 text-indigo-500" />
          NLP Pipeline
        </h2>
        <div className="flex flex-wrap items-center gap-2">
          {PIPELINE_STEPS.map((s, i) => (
            <div key={s.step} className="flex items-center gap-2">
              <div className="text-center">
                <div className={`${s.color} text-white rounded-lg px-3 py-2 text-sm font-semibold min-w-[120px]`}>
                  <div>{s.label}</div>
                  <div className="text-xs font-normal opacity-80 mt-0.5">{s.sub}</div>
                </div>
              </div>
              {i < PIPELINE_STEPS.length - 1 && (
                <ChevronRight className="w-5 h-5 text-slate-300 flex-shrink-0" />
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Technology Stack */}
      <div className="bg-white rounded-xl shadow-sm p-6 mb-8">
        <h2 className="text-xl font-bold text-slate-800 mb-4">Technology Stack</h2>
        {['Frontend', 'Backend', 'ML', 'NLP', 'Dataset'].map(cat => (
          <div key={cat} className="mb-3">
            <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">{cat}</div>
            <div className="flex flex-wrap gap-2">
              {TECH_STACK.filter(t => t.cat === cat).map(t => (
                <span key={t.label} className={`${t.color} text-xs font-medium px-2.5 py-1 rounded-full`}>{t.label}</span>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Architecture Diagram */}
      <div className="bg-white rounded-xl shadow-sm p-6 mb-8">
        <h2 className="text-xl font-bold text-slate-800 mb-4">Architecture</h2>
        <div className="bg-slate-900 rounded-xl p-6 font-mono text-sm text-slate-300 overflow-x-auto">
          <pre>{`
  ┌─────────────────────────┐
  │   React Frontend         │  http://localhost:5173
  │   (Vite + Tailwind)      │
  └──────────┬──────────────┘
             │ HTTP REST (Axios)
             ▼
  ┌─────────────────────────┐
  │   FastAPI Backend        │  http://localhost:8000
  │   Python 3.12            │
  └──────┬──────────┬────────┘
         │          │
   ┌─────▼──┐  ┌───▼──────────┐
   │ TF-IDF │  │  DistilBERT  │
   │  + LR  │  │  Transformer │
   └─────┬──┘  └───┬──────────┘
         └────┬────┘
              ▼
     Post-processing
     (confidence, labels,
      keywords, formatting)
              │
              ▼
      JSON Response → UI
          `.trim()}</pre>
        </div>
      </div>

      {/* Future Work */}
      <div className="bg-white rounded-xl shadow-sm p-6">
        <h2 className="text-xl font-bold text-slate-800 mb-4 flex items-center gap-2">
          <FlaskConical className="w-5 h-5 text-indigo-500" />
          Future Work
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          {FUTURE_WORK.map((item, i) => (
            <div key={i} className="flex items-start gap-2 text-sm text-slate-600 p-2 rounded-lg hover:bg-slate-50">
              <span className="text-indigo-400 font-bold mt-0.5">→</span>
              {item}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
