import { useState, useEffect } from 'react'
import { Loader2, Trash2, AlertTriangle, Clock, Hash, AlignLeft, Zap, BrainCircuit } from 'lucide-react'
import toast from 'react-hot-toast'
import { predictNews } from '../services/api'
import CategoryBadge from '../components/CategoryBadge'
import ProbabilityBar from '../components/ProbabilityBar'
import ConfidenceGauge from '../components/ConfidenceGauge'

const EXAMPLE_TEXTS = {
  World: {
    headline: 'UN Security Council Holds Emergency Session Over Border Conflict',
    article:
      'The United Nations Security Council held an emergency session on Monday to address the escalating conflict at the border between the two nations. Diplomatic negotiations are underway as world leaders call for an immediate ceasefire and humanitarian aid. The Secretary-General urged all parties to exercise restraint and engage in dialogue to prevent further casualties. Several countries have offered to mediate peace talks while international aid agencies prepare emergency relief supplies.',
  },
  Sports: {
    headline: 'Manchester City Defeat Arsenal 3-1 in Premier League Thriller',
    article:
      "Manchester City defeated Arsenal 3-1 in a breathtaking Premier League encounter at the Etihad Stadium. Erling Haaland scored a brace while Kevin De Bruyne added a stunning long-range goal. The result extends City's lead at the top of the table to five points. Arsenal manager Mikel Arteta acknowledged the defeat but remained confident about his team's title chances for the rest of the season. The match drew a record TV audience with over 12 million viewers.",
  },
  Business: {
    headline: 'Apple Reports Record $119.6B Revenue, Announces $110B Buyback',
    article:
      'Apple Inc. reported record quarterly revenues of $119.6 billion, surpassing analyst expectations. Strong iPhone 15 sales and a rapidly growing services segment including the App Store, Apple Music, and iCloud drove the impressive results. The company also announced a $110 billion share buyback program. CEO Tim Cook cited robust demand across all geographic regions and expressed optimism about the upcoming product launches in the next fiscal quarter.',
  },
  'Sci/Tech': {
    headline: 'MIT AI Algorithm Detects Early-Stage Cancer with 94% Accuracy',
    article:
      'Researchers at MIT have developed a new artificial intelligence algorithm capable of detecting early-stage cancer from routine blood tests with 94% accuracy. The deep learning model was trained on over one million patient records and could revolutionize early diagnosis in oncology. Clinical trials are expected to begin within the next six months at major hospitals across the United States. The team published their findings in the journal Nature Medicine.',
  },
}

const CATEGORIES = ['World', 'Sports', 'Business', 'Sci/Tech']

const MODEL_OPTIONS = [
  { value: 'tfidf_lr', label: 'TF-IDF + LR (Fast)', icon: <Zap className="w-4 h-4" /> },
  { value: 'distilbert', label: 'DistilBERT (Advanced)', icon: <BrainCircuit className="w-4 h-4" /> },
]

export default function ClassifyPage() {
  const [headline, setHeadline] = useState('')
  const [article, setArticle] = useState('')
  const [modelType, setModelType] = useState('tfidf_lr')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)

  const wordCount = article.trim() ? article.trim().split(/\s+/).length : 0
  const charCount = article.length

  const loadExample = (category) => {
    setHeadline(EXAMPLE_TEXTS[category].headline)
    setArticle(EXAMPLE_TEXTS[category].article)
    setResult(null)
    toast.success(`Loaded ${category} example`)
  }

  const handleClear = () => {
    setHeadline('')
    setArticle('')
    setResult(null)
  }

  const handleClassify = async () => {
    if (!article.trim()) {
      toast.error('Please enter some article text to classify.')
      return
    }
    setLoading(true)
    try {
      const data = await predictNews(headline, article, modelType)
      setResult(data)
      toast.success(`Classified as ${data.predicted_category}`)
    } catch (err) {
      toast.error(err.message || 'Classification failed. Is the backend running?')
    } finally {
      setLoading(false)
    }
  }

  const exampleBtnColor = {
    World: 'bg-blue-50 text-blue-700 border-blue-200 hover:bg-blue-100',
    Sports: 'bg-green-50 text-green-700 border-green-200 hover:bg-green-100',
    Business: 'bg-orange-50 text-orange-700 border-orange-200 hover:bg-orange-100',
    'Sci/Tech': 'bg-purple-50 text-purple-700 border-purple-200 hover:bg-purple-100',
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-slate-900 mb-1">News Classifier</h1>
        <p className="text-slate-500">Enter a news article and select a model to predict its topic category.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* LEFT: Input Panel */}
        <div className="space-y-5">
          <div className="card">
            {/* Headline */}
            <div className="mb-4">
              <label className="block text-sm font-semibold text-slate-700 mb-1.5">
                Headline <span className="text-slate-400 font-normal">(optional)</span>
              </label>
              <input
                type="text"
                className="input-field"
                placeholder="Enter news headline..."
                value={headline}
                onChange={(e) => setHeadline(e.target.value)}
              />
            </div>

            {/* Article */}
            <div className="mb-4">
              <label className="block text-sm font-semibold text-slate-700 mb-1.5">
                Article Text <span className="text-red-500">*</span>
              </label>
              <textarea
                rows={10}
                className="input-field resize-none font-mono text-xs leading-relaxed"
                placeholder="Paste your news article here..."
                value={article}
                onChange={(e) => setArticle(e.target.value)}
              />
              {/* Live stats */}
              <div className="flex gap-4 mt-1.5 text-xs text-slate-400">
                <span className="flex items-center gap-1">
                  <Hash className="w-3 h-3" /> {wordCount} words
                </span>
                <span className="flex items-center gap-1">
                  <AlignLeft className="w-3 h-3" /> {charCount} chars
                </span>
              </div>
            </div>

            {/* Example buttons */}
            <div className="mb-4">
              <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">Load Example</p>
              <div className="flex flex-wrap gap-2">
                {CATEGORIES.map((cat) => (
                  <button
                    key={cat}
                    onClick={() => loadExample(cat)}
                    className={`text-xs border rounded-md px-3 py-1.5 font-medium transition-colors ${exampleBtnColor[cat]}`}
                  >
                    {cat} Example
                  </button>
                ))}
              </div>
            </div>

            {/* Model selector */}
            <div className="mb-5">
              <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">Select Model</p>
              <div className="grid grid-cols-2 gap-2">
                {MODEL_OPTIONS.map((opt) => (
                  <button
                    key={opt.value}
                    onClick={() => setModelType(opt.value)}
                    className={`flex items-center gap-2 px-4 py-2.5 rounded-lg border text-sm font-medium transition-all ${
                      modelType === opt.value
                        ? 'bg-indigo-600 text-white border-indigo-600 shadow-sm'
                        : 'bg-white text-slate-600 border-slate-300 hover:border-indigo-400 hover:text-indigo-600'
                    }`}
                  >
                    {opt.icon}
                    {opt.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Actions */}
            <div className="flex gap-3">
              <button
                onClick={handleClear}
                className="btn-secondary flex items-center gap-2 flex-1"
              >
                <Trash2 className="w-4 h-4" />
                Clear
              </button>
              <button
                onClick={handleClassify}
                disabled={loading}
                className="btn-primary flex items-center justify-center gap-2 flex-2 flex-1"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Classifying...
                  </>
                ) : (
                  <>
                    <BrainCircuit className="w-4 h-4" />
                    Classify News
                  </>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* RIGHT: Results Panel */}
        <div className="space-y-5">
          {!result && !loading && (
            <div className="card flex flex-col items-center justify-center text-center py-16 text-slate-400">
              <BrainCircuit className="w-12 h-12 mb-4 text-slate-200" />
              <p className="font-semibold text-slate-500 mb-1">No prediction yet</p>
              <p className="text-sm">Enter article text and click "Classify News"</p>
            </div>
          )}

          {loading && (
            <div className="card flex flex-col items-center justify-center text-center py-16">
              <Loader2 className="w-10 h-10 text-indigo-600 animate-spin mb-4" />
              <p className="text-slate-500 font-medium">Running inference...</p>
              <p className="text-xs text-slate-400 mt-1">
                {modelType === 'distilbert' ? 'DistilBERT may take a few seconds' : 'TF-IDF model is fast'}
              </p>
            </div>
          )}

          {result && !loading && (
            <>
              {/* Main prediction card */}
              <div className="card animate-fade-in">
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Predicted Category</p>
                    <CategoryBadge category={result.predicted_category} size="xl" />
                  </div>
                  <ConfidenceGauge confidence={result.confidence} />
                </div>

                {/* Confidence bar */}
                <div className="mt-4">
                  <div className="flex justify-between text-xs text-slate-500 mb-1">
                    <span>Confidence</span>
                    <span className="font-bold">{Math.round(result.confidence * 100)}%</span>
                  </div>
                  <div className="h-3 bg-slate-200 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full progress-bar-animated ${
                        result.confidence >= 0.7
                          ? 'bg-green-500'
                          : result.confidence >= 0.5
                          ? 'bg-yellow-500'
                          : 'bg-red-500'
                      }`}
                      style={{ width: `${result.confidence * 100}%` }}
                    />
                  </div>
                </div>

                {result.confidence < 0.5 && (
                  <div className="mt-3 flex items-start gap-2 p-3 bg-red-50 border border-red-200 rounded-lg text-red-700">
                    <AlertTriangle className="w-4 h-4 flex-shrink-0 mt-0.5" />
                    <p className="text-xs">
                      Low confidence ({Math.round(result.confidence * 100)}%). The article may be ambiguous or contain mixed topics.
                    </p>
                  </div>
                )}
              </div>

              {/* Probability distribution */}
              <div className="card animate-fade-in">
                <h3 className="text-sm font-semibold text-slate-700 mb-4 flex items-center gap-2">
                  <span className="w-2 h-2 bg-indigo-500 rounded-full" />
                  Probability Distribution
                </h3>
                <div className="space-y-2">
                  {result.probabilities &&
                    Object.entries(result.probabilities)
                      .sort(([, a], [, b]) => b - a)
                      .map(([cat, prob]) => (
                        <ProbabilityBar
                          key={cat}
                          category={cat}
                          probability={prob}
                          isTop={cat === result.predicted_category}
                        />
                      ))}
                </div>
              </div>

              {/* Keywords */}
              {result.top_keywords && result.top_keywords.length > 0 && (
                <div className="card animate-fade-in">
                  <h3 className="text-sm font-semibold text-slate-700 mb-3 flex items-center gap-2">
                    <span className="w-2 h-2 bg-violet-500 rounded-full" />
                    Detected Keywords
                  </h3>
                  <div className="flex flex-wrap gap-2">
                    {result.top_keywords.map((kw) => (
                      <span
                        key={kw}
                        className="px-2.5 py-1 bg-indigo-50 text-indigo-700 border border-indigo-200 rounded-full text-xs font-medium"
                      >
                        {kw}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Processing info */}
              <div className="card animate-fade-in">
                <h3 className="text-sm font-semibold text-slate-700 mb-3 flex items-center gap-2">
                  <span className="w-2 h-2 bg-slate-400 rounded-full" />
                  Processing Info
                </h3>
                <div className="grid grid-cols-3 gap-4 text-center">
                  <div className="bg-slate-50 rounded-lg p-3">
                    <p className="text-xs text-slate-400 mb-1">Model</p>
                    <p className="text-sm font-bold text-slate-700">
                      {result.model_used === 'distilbert' ? 'DistilBERT' : 'TF-IDF + LR'}
                    </p>
                  </div>
                  <div className="bg-slate-50 rounded-lg p-3">
                    <p className="text-xs text-slate-400 mb-1">Time</p>
                    <p className="text-sm font-bold text-slate-700 flex items-center justify-center gap-1">
                      <Clock className="w-3 h-3" />
                      {result.processing_time_ms != null
                        ? `${result.processing_time_ms}ms`
                        : '—'}
                    </p>
                  </div>
                  <div className="bg-slate-50 rounded-lg p-3">
                    <p className="text-xs text-slate-400 mb-1">Words</p>
                    <p className="text-sm font-bold text-slate-700">{result.word_count || wordCount}</p>
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
