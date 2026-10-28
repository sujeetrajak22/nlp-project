import { useState, useEffect } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer
} from 'recharts'
import { getModelMetrics } from '../services/api'
import LoadingSpinner from '../components/LoadingSpinner'
import StatCard from '../components/StatCard'
import { TrendingUp, Clock, Target, AlertCircle, CheckCircle2, Zap, BrainCircuit } from 'lucide-react'

const CATEGORY_COLORS = {
  World: '#3b82f6',
  Sports: '#22c55e',
  Business: '#f97316',
  'Sci/Tech': '#a855f7',
}

function MetricBar({ label, value, color = 'indigo' }) {
  const pct = Math.round((value || 0) * 100)
  const colorMap = {
    indigo: 'bg-indigo-500',
    green: 'bg-green-500',
    orange: 'bg-orange-500',
    purple: 'bg-purple-500',
  }
  return (
    <div className="mb-3">
      <div className="flex justify-between mb-1">
        <span className="text-sm font-medium text-slate-600">{label}</span>
        <span className="text-sm font-bold text-slate-800">{pct}%</span>
      </div>
      <div className="w-full bg-slate-200 rounded-full h-2">
        <div
          className={`${colorMap[color] || 'bg-indigo-500'} h-2 rounded-full transition-all duration-700`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}

function ConfusionMatrix({ matrix, categories }) {
  if (!matrix || !categories) return null
  const maxVal = Math.max(...matrix.flat())
  return (
    <div className="overflow-x-auto">
      <table className="text-xs border-collapse w-full">
        <thead>
          <tr>
            <th className="p-2 text-slate-500 font-normal text-right">Actual ↓ / Pred →</th>
            {categories.map(c => (
              <th key={c} className="p-2 font-semibold text-slate-700 text-center min-w-[80px]">{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              <td className="p-2 font-semibold text-slate-700 text-right">{categories[i]}</td>
              {row.map((val, j) => {
                const intensity = maxVal > 0 ? val / maxVal : 0
                const isCorrect = i === j
                const bg = isCorrect
                  ? `rgba(99,102,241,${0.15 + intensity * 0.75})`
                  : `rgba(239,68,68,${intensity * 0.5})`
                return (
                  <td
                    key={j}
                    className="p-2 text-center font-bold border border-slate-100"
                    style={{ backgroundColor: bg }}
                  >
                    {val?.toLocaleString() ?? 0}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="flex gap-4 mt-2 text-xs text-slate-500">
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-3 rounded" style={{background:'rgba(99,102,241,0.7)'}}/> Correct</span>
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-3 rounded" style={{background:'rgba(239,68,68,0.4)'}}/> Misclassified</span>
      </div>
    </div>
  )
}

function ModelCard({ title, icon, metrics, color }) {
  if (!metrics) return null
  return (
    <div className={`bg-white rounded-xl shadow-sm border-2 ${color} p-6`}>
      <div className="flex items-center gap-3 mb-4">
        <div className="text-2xl">{icon}</div>
        <h3 className="font-bold text-slate-800 text-lg">{title}</h3>
      </div>
      <MetricBar label="Accuracy" value={metrics.accuracy} color="indigo" />
      <MetricBar label="Macro F1" value={metrics.macro_f1} color="green" />
      <MetricBar label="Weighted F1" value={metrics.weighted_f1} color="orange" />
      <div className="mt-4 flex items-center gap-2 text-sm text-slate-500">
        <Clock className="w-4 h-4" />
        <span>Inference: <b className="text-slate-700">{metrics.inference_time_ms?.toFixed(2) ?? '—'} ms/sample</b></span>
      </div>
    </div>
  )
}

export default function PerformancePage() {
  const [metrics, setMetrics] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    getModelMetrics()
      .then(data => setMetrics(data))
      .catch(err => setError(err.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <LoadingSpinner size="lg" label="Loading model metrics…" />
      </div>
    )
  }

  if (error || metrics?.status === 'not_generated' || metrics?.status === 'error') {
    return (
      <div className="max-w-3xl mx-auto px-4 py-20 text-center">
        <AlertCircle className="w-16 h-16 text-amber-400 mx-auto mb-4" />
        <h2 className="text-2xl font-bold text-slate-800 mb-3">Model Metrics Not Available</h2>
        <p className="text-slate-500 mb-6">
          {metrics?.message || 'Run the training pipeline to generate evaluation results.'}
        </p>
        <div className="bg-slate-900 rounded-xl p-6 text-left text-sm font-mono text-slate-300 space-y-1">
          <p><span className="text-green-400"># Step 1 – download data</span></p>
          <p>python scripts/download_dataset.py</p>
          <p><span className="text-green-400"># Step 2 – preprocess</span></p>
          <p>python scripts/preprocess.py</p>
          <p><span className="text-green-400"># Step 3 – train baseline</span></p>
          <p>python scripts/train_baseline.py</p>
          <p><span className="text-green-400"># Step 4 – evaluate</span></p>
          <p>python scripts/evaluate.py</p>
        </div>
      </div>
    )
  }

  const tfidf = metrics?.tfidf_lr
  const bert = metrics?.distilbert

  // Comparison chart data
  const comparisonData = ['accuracy', 'macro_f1', 'weighted_f1'].map(key => ({
    metric: key === 'accuracy' ? 'Accuracy' : key === 'macro_f1' ? 'Macro F1' : 'Weighted F1',
    'TF-IDF + LR': tfidf ? Math.round((tfidf[key] || 0) * 1000) / 10 : 0,
    DistilBERT: bert ? Math.round((bert[key] || 0) * 1000) / 10 : 0,
  }))

  // Per-class F1 chart
  const categories = tfidf?.categories || ['World', 'Sports', 'Business', 'Sci/Tech']
  const perClassData = categories.map(cat => ({
    category: cat,
    'TF-IDF + LR': tfidf?.classification_report?.[cat]?.['f1-score']
      ? Math.round(tfidf.classification_report[cat]['f1-score'] * 1000) / 10
      : 0,
    DistilBERT: bert?.classification_report?.[cat]?.['f1-score']
      ? Math.round(bert.classification_report[cat]['f1-score'] * 1000) / 10
      : 0,
  }))

  return (
    <div className="max-w-7xl mx-auto px-4 py-10">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-slate-800 mb-2">Model Performance</h1>
        <p className="text-slate-500 text-lg">
          Comparison of TF-IDF + Logistic Regression vs DistilBERT on the AG News test set (7,600 samples).
        </p>
      </div>

      {/* Quick-stat cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-8">
        <StatCard
          icon={<Target className="w-5 h-5 text-indigo-500" />}
          label="TF-IDF Accuracy"
          value={tfidf ? `${(tfidf.accuracy * 100).toFixed(1)}%` : '—'}
          bg="bg-indigo-50"
        />
        <StatCard
          icon={<TrendingUp className="w-5 h-5 text-purple-500" />}
          label="DistilBERT Accuracy"
          value={bert ? `${(bert.accuracy * 100).toFixed(1)}%` : 'Not trained'}
          bg="bg-purple-50"
        />
        <StatCard
          icon={<Zap className="w-5 h-5 text-amber-500" />}
          label="TF-IDF Inference"
          value={tfidf ? `${tfidf.inference_time_ms?.toFixed(1)} ms` : '—'}
          bg="bg-amber-50"
        />
        <StatCard
          icon={<Clock className="w-5 h-5 text-blue-500" />}
          label="DistilBERT Inference"
          value={bert ? `${bert.inference_time_ms?.toFixed(0)} ms` : '—'}
          bg="bg-blue-50"
        />
      </div>

      {/* Model cards side-by-side */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        <ModelCard title="TF-IDF + Logistic Regression" icon="⚡" metrics={tfidf} color="border-indigo-200" />
        <ModelCard title="DistilBERT (Transformer)" icon="🤖" metrics={bert} color="border-purple-200" />
      </div>

      {/* Model comparison bar chart */}
      <div className="bg-white rounded-xl shadow-sm p-6 mb-8">
        <h2 className="text-lg font-bold text-slate-800 mb-1">Overall Metrics Comparison (%)</h2>
        <p className="text-sm text-slate-500 mb-4">Accuracy, Macro F1 and Weighted F1 on AG News test set</p>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={comparisonData} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis dataKey="metric" tick={{ fontSize: 13 }} />
            <YAxis domain={[80, 100]} tick={{ fontSize: 12 }} unit="%" />
            <Tooltip formatter={(v) => `${v}%`} />
            <Legend />
            <Bar dataKey="TF-IDF + LR" fill="#6366f1" radius={[4,4,0,0]} />
            {bert && <Bar dataKey="DistilBERT" fill="#a855f7" radius={[4,4,0,0]} />}
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Per-class F1 */}
      <div className="bg-white rounded-xl shadow-sm p-6 mb-8">
        <h2 className="text-lg font-bold text-slate-800 mb-1">Per-Class F1 Score (%)</h2>
        <p className="text-sm text-slate-500 mb-4">F1 score for each AG News category</p>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={perClassData} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis dataKey="category" tick={{ fontSize: 13 }} />
            <YAxis domain={[80, 100]} tick={{ fontSize: 12 }} unit="%" />
            <Tooltip formatter={(v) => `${v}%`} />
            <Legend />
            <Bar dataKey="TF-IDF + LR" fill="#6366f1" radius={[4,4,0,0]} />
            {bert && <Bar dataKey="DistilBERT" fill="#a855f7" radius={[4,4,0,0]} />}
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Confusion matrices */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        {tfidf?.confusion_matrix && (
          <div className="bg-white rounded-xl shadow-sm p-6">
            <h2 className="text-lg font-bold text-slate-800 mb-4">Confusion Matrix — TF-IDF + LR</h2>
            <ConfusionMatrix matrix={tfidf.confusion_matrix} categories={categories} />
          </div>
        )}
        {bert?.confusion_matrix && (
          <div className="bg-white rounded-xl shadow-sm p-6">
            <h2 className="text-lg font-bold text-slate-800 mb-4">Confusion Matrix — DistilBERT</h2>
            <ConfusionMatrix matrix={bert.confusion_matrix} categories={categories} />
          </div>
        )}
      </div>

      {/* Per-class classification report table */}
      {tfidf?.classification_report && (
        <div className="bg-white rounded-xl shadow-sm p-6 mb-8">
          <h2 className="text-lg font-bold text-slate-800 mb-4">Classification Report — TF-IDF + LR</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-200">
                  <th className="text-left py-2 px-3 text-slate-600 font-semibold">Category</th>
                  <th className="text-right py-2 px-3 text-slate-600 font-semibold">Precision</th>
                  <th className="text-right py-2 px-3 text-slate-600 font-semibold">Recall</th>
                  <th className="text-right py-2 px-3 text-slate-600 font-semibold">F1-Score</th>
                  <th className="text-right py-2 px-3 text-slate-600 font-semibold">Support</th>
                </tr>
              </thead>
              <tbody>
                {categories.map((cat) => {
                  const r = tfidf.classification_report[cat]
                  if (!r) return null
                  return (
                    <tr key={cat} className="border-b border-slate-50 hover:bg-slate-50">
                      <td className="py-2 px-3 font-medium text-slate-800">{cat}</td>
                      <td className="py-2 px-3 text-right text-slate-700">{(r.precision * 100).toFixed(1)}%</td>
                      <td className="py-2 px-3 text-right text-slate-700">{(r.recall * 100).toFixed(1)}%</td>
                      <td className="py-2 px-3 text-right font-semibold text-indigo-600">{(r['f1-score'] * 100).toFixed(1)}%</td>
                      <td className="py-2 px-3 text-right text-slate-500">{r.support?.toLocaleString()}</td>
                    </tr>
                  )
                })}
                {tfidf.classification_report['macro avg'] && (
                  <tr className="border-t-2 border-slate-200 bg-slate-50 font-semibold">
                    <td className="py-2 px-3 text-slate-700">Macro Avg</td>
                    <td className="py-2 px-3 text-right">{(tfidf.classification_report['macro avg'].precision * 100).toFixed(1)}%</td>
                    <td className="py-2 px-3 text-right">{(tfidf.classification_report['macro avg'].recall * 100).toFixed(1)}%</td>
                    <td className="py-2 px-3 text-right text-indigo-600">{(tfidf.classification_report['macro avg']['f1-score'] * 100).toFixed(1)}%</td>
                    <td className="py-2 px-3 text-right text-slate-500">—</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Model architecture comparison */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white rounded-xl shadow-sm p-6 border-l-4 border-indigo-500">
          <div className="flex items-center gap-2 mb-3">
            <Zap className="w-5 h-5 text-indigo-500" />
            <h3 className="font-bold text-slate-800">TF-IDF + Logistic Regression</h3>
          </div>
          <div className="space-y-2 text-sm">
            <div className="flex gap-2"><CheckCircle2 className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0"/><span>Fast inference (&lt;5ms/sample)</span></div>
            <div className="flex gap-2"><CheckCircle2 className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0"/><span>Interpretable (feature importance)</span></div>
            <div className="flex gap-2"><CheckCircle2 className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0"/><span>Low memory footprint</span></div>
            <div className="flex gap-2"><CheckCircle2 className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0"/><span>Trains in minutes on CPU</span></div>
            <div className="flex gap-2"><AlertCircle className="w-4 h-4 text-amber-500 mt-0.5 flex-shrink-0"/><span>No contextual understanding</span></div>
            <div className="flex gap-2"><AlertCircle className="w-4 h-4 text-amber-500 mt-0.5 flex-shrink-0"/><span>Bag-of-words assumption</span></div>
          </div>
        </div>
        <div className="bg-white rounded-xl shadow-sm p-6 border-l-4 border-purple-500">
          <div className="flex items-center gap-2 mb-3">
            <BrainCircuit className="w-5 h-5 text-purple-500" />
            <h3 className="font-bold text-slate-800">DistilBERT (Transformer)</h3>
          </div>
          <div className="space-y-2 text-sm">
            <div className="flex gap-2"><CheckCircle2 className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0"/><span>Deep contextual understanding</span></div>
            <div className="flex gap-2"><CheckCircle2 className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0"/><span>Handles negation, irony</span></div>
            <div className="flex gap-2"><CheckCircle2 className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0"/><span>State-of-the-art accuracy</span></div>
            <div className="flex gap-2"><CheckCircle2 className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0"/><span>Pre-trained on large corpus</span></div>
            <div className="flex gap-2"><AlertCircle className="w-4 h-4 text-amber-500 mt-0.5 flex-shrink-0"/><span>Slower inference (~200ms CPU)</span></div>
            <div className="flex gap-2"><AlertCircle className="w-4 h-4 text-amber-500 mt-0.5 flex-shrink-0"/><span>Needs GPU for fast training</span></div>
          </div>
        </div>
      </div>
    </div>
  )
}
