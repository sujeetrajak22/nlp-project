import { useState } from 'react'
import { Loader2, ArrowDown, Info } from 'lucide-react'
import toast from 'react-hot-toast'
import { preprocessText } from '../services/api'

const DEFAULT_TEXT =
  "The Government's new healthcare bill has been passed by the Senate, including several controversial provisions about Medicare funding and pharmaceutical pricing regulations."

const STAGES = [
  { key: 'raw', label: '1. Raw Text', color: 'bg-slate-500', desc: 'Original unprocessed input' },
  { key: 'html_removed', label: '2. HTML/URL Removed', color: 'bg-blue-500', desc: 'Strip HTML tags and URLs' },
  { key: 'lowercased', label: '3. Lowercased', color: 'bg-indigo-500', desc: 'Normalize to lowercase' },
  { key: 'special_chars_removed', label: '4. Special Chars Removed', color: 'bg-violet-500', desc: 'Remove punctuation & symbols' },
  { key: 'tokenized', label: '5. Tokenized', color: 'bg-purple-500', desc: 'Split into word tokens' },
  { key: 'stopwords_removed', label: '6. Stopwords Removed', color: 'bg-fuchsia-500', desc: 'Filter common stopwords' },
  { key: 'lemmatized', label: '7. Lemmatized', color: 'bg-pink-500', desc: 'Reduce words to base form' },
]

function StageCard({ stage, data, isLoaded }) {
  if (!isLoaded) return null
  const text = Array.isArray(data) ? data.join(' ') : (data || '')
  const words = text.trim() ? text.trim().split(/\s+/) : []

  return (
    <div className="relative">
      <div className="card hover:shadow-md transition-shadow border-l-4" style={{ borderLeftColor: stage.color.replace('bg-', '').includes('-') ? undefined : '#6366f1' }}>
        <div className="flex items-start justify-between mb-2">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className={`w-2.5 h-2.5 rounded-full ${stage.color} flex-shrink-0`} />
              <h3 className="text-sm font-bold text-slate-800">{stage.label}</h3>
            </div>
            <p className="text-xs text-slate-400">{stage.desc}</p>
          </div>
          <div className="flex gap-3 text-xs text-slate-400 flex-shrink-0 ml-4">
            <span className="bg-slate-100 px-2 py-0.5 rounded">{words.length} tokens</span>
            <span className="bg-slate-100 px-2 py-0.5 rounded">{text.length} chars</span>
          </div>
        </div>
        <div className="bg-slate-50 rounded-lg p-3 font-mono text-xs text-slate-700 leading-relaxed max-h-24 overflow-y-auto break-words">
          {text.length > 300 ? text.slice(0, 300) + '…' : text || <span className="text-slate-300 italic">empty</span>}
        </div>
      </div>
      <div className="flex justify-center my-2">
        <ArrowDown className="w-4 h-4 text-slate-300" />
      </div>
    </div>
  )
}

export default function PreprocessingPage() {
  const [inputText, setInputText] = useState(DEFAULT_TEXT)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  const handleRun = async () => {
    if (!inputText.trim()) {
      toast.error('Please enter some text to preprocess.')
      return
    }
    setLoading(true)
    try {
      const data = await preprocessText(inputText)
      setResult(data)
      toast.success('Preprocessing complete!')
    } catch (err) {
      toast.error(err.message || 'Preprocessing failed. Is the backend running?')
    } finally {
      setLoading(false)
    }
  }

  // Build stage data: if API returns steps, use them; otherwise simulate
  const getStageData = (key) => {
    if (!result) return null
    if (result.stages && result.stages[key] !== undefined) return result.stages[key]
    // Fallback keys
    const fallback = {
      raw: result.raw || inputText,
      html_removed: result.html_removed || result.raw || inputText,
      lowercased: result.lowercased,
      special_chars_removed: result.special_chars_removed || result.cleaned,
      tokenized: result.tokenized || result.tokens,
      stopwords_removed: result.stopwords_removed || result.filtered_tokens,
      lemmatized: result.lemmatized || result.lemmatized_tokens || result.tfidf_preprocessed,
    }
    return fallback[key]
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-slate-900 mb-1">Preprocessing Visualizer</h1>
        <p className="text-slate-500">See step-by-step how raw text is transformed through the NLP pipeline.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Input section */}
        <div className="lg:col-span-1 space-y-5">
          <div className="card">
            <label className="block text-sm font-semibold text-slate-700 mb-2">Input Text</label>
            <textarea
              rows={8}
              className="input-field resize-none text-sm mb-4"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder="Enter text to preprocess..."
            />
            <button
              onClick={handleRun}
              disabled={loading}
              className="btn-primary w-full flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Processing...
                </>
              ) : (
                'Run Preprocessing'
              )}
            </button>
          </div>

          {/* Final outputs comparison */}
          {result && (
            <div className="card animate-fade-in">
              <h3 className="text-sm font-bold text-slate-800 mb-3 flex items-center gap-2">
                <Info className="w-4 h-4 text-indigo-500" />
                Final Outputs Comparison
              </h3>
              <div className="space-y-4">
                <div>
                  <p className="text-xs font-semibold text-indigo-600 mb-1">TF-IDF Preprocessed</p>
                  <div className="bg-indigo-50 rounded-lg p-3 font-mono text-xs text-indigo-800 break-words max-h-28 overflow-y-auto">
                    {(() => {
                      const t = result.tfidf_preprocessed || result.tfidf_text || ''
                      return t.length > 200 ? t.slice(0, 200) + '…' : t || <span className="italic text-indigo-300">not available</span>
                    })()}
                  </div>
                </div>
                <div>
                  <p className="text-xs font-semibold text-purple-600 mb-1">Transformer Preprocessed</p>
                  <div className="bg-purple-50 rounded-lg p-3 font-mono text-xs text-purple-800 break-words max-h-28 overflow-y-auto">
                    {(() => {
                      const t = result.transformer_preprocessed || result.transformer_text || ''
                      return t.length > 200 ? t.slice(0, 200) + '…' : t || <span className="italic text-purple-300">not available</span>
                    })()}
                  </div>
                </div>
              </div>
              <div className="mt-4 p-3 bg-amber-50 border border-amber-200 rounded-lg">
                <p className="text-xs text-amber-700">
                  <strong>Why different?</strong> Transformers like DistilBERT use their own built-in tokenizer (WordPiece). Heavy preprocessing like stemming/lemmatization can hurt performance, so only minimal cleaning (lowercasing) is applied.
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Pipeline stages */}
        <div className="lg:col-span-2">
          {!result && !loading && (
            <div className="card flex flex-col items-center justify-center text-center py-20 text-slate-400">
              <div className="text-5xl mb-4">🔬</div>
              <p className="font-semibold text-slate-500 mb-1">No preprocessing run yet</p>
              <p className="text-sm">Enter text and click "Run Preprocessing"</p>
            </div>
          )}

          {loading && (
            <div className="card flex flex-col items-center justify-center text-center py-20">
              <Loader2 className="w-10 h-10 text-indigo-600 animate-spin mb-4" />
              <p className="text-slate-500">Running pipeline...</p>
            </div>
          )}

          {result && !loading && (
            <div className="animate-slide-up">
              {STAGES.map((stage) => {
                const data = getStageData(stage.key)
                return (
                  <StageCard
                    key={stage.key}
                    stage={stage}
                    data={data !== undefined && data !== null ? data : (stage.key === 'raw' ? inputText : null)}
                    isLoaded={true}
                  />
                )
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
