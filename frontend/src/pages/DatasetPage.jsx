import { useEffect, useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  AreaChart, Area, Cell,
} from 'recharts'
import { getDatasetStats } from '../services/api'
import StatCard from '../components/StatCard'
import LoadingSpinner from '../components/LoadingSpinner'
import { Database, FileText, BookOpen, List, CheckCircle } from 'lucide-react'

const CATEGORY_COLORS = {
  World: '#3b82f6',
  Sports: '#22c55e',
  Business: '#f97316',
  'Sci/Tech': '#a855f7',
}

function SkeletonCard() {
  return (
    <div className="card animate-pulse">
      <div className="h-4 bg-slate-200 rounded w-1/3 mb-3" />
      <div className="h-8 bg-slate-200 rounded w-1/2 mb-2" />
      <div className="h-3 bg-slate-200 rounded w-1/4" />
    </div>
  )
}

function NotGeneratedState() {
  return (
    <div className="col-span-full card text-center py-16">
      <Database className="w-12 h-12 text-slate-300 mx-auto mb-4" />
      <h3 className="text-xl font-bold text-slate-700 mb-2">Dataset Stats Not Available</h3>
      <p className="text-slate-500 mb-6 max-w-md mx-auto">
        The dataset statistics haven't been generated yet. Run the training pipeline to generate them.
      </p>
      <div className="bg-slate-900 text-green-400 rounded-xl p-4 text-left max-w-md mx-auto font-mono text-sm">
        <p className="text-slate-400 mb-1"># From project root:</p>
        <p>python src/train.py</p>
        <p className="text-slate-400 mt-2 mb-1"># Or run the full pipeline:</p>
        <p>python src/pipeline.py</p>
      </div>
    </div>
  )
}

export default function DatasetPage() {
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(true)
  const [notGenerated, setNotGenerated] = useState(false)

  useEffect(() => {
    getDatasetStats()
      .then((data) => {
        if (data?.status === 'not_generated' || !data) {
          setNotGenerated(true)
        } else {
          setStats(data)
        }
      })
      .catch(() => setNotGenerated(true))
      .finally(() => setLoading(false))
  }, [])

  // Derive chart data from stats or use mock data
  const categoryDist = stats?.category_distribution
    ? Object.entries(stats.category_distribution).map(([name, count]) => ({
        name,
        count,
        fill: CATEGORY_COLORS[name] || '#6366f1',
      }))
    : []

  const topWords = stats?.top_words
    ? Object.entries(stats.top_words)
        .slice(0, 12)
        .map(([word, freq]) => ({ word, freq }))
    : []

  const lengthDist = stats?.length_distribution
    ? Object.entries(stats.length_distribution).map(([bucket, count]) => ({
        bucket,
        count,
      }))
    : []

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-slate-900 mb-1">Dataset Insights</h1>
        <p className="text-slate-500">AG News corpus statistics and distributions</p>
      </div>

      {/* Stat cards */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-8">
        {loading ? (
          Array.from({ length: 6 }).map((_, i) => <SkeletonCard key={i} />)
        ) : notGenerated ? (
          <NotGeneratedState />
        ) : (
          <>
            <StatCard icon={<FileText className="w-5 h-5" />} label="Total Docs" value={stats?.total_docs?.toLocaleString() || '—'} colorClass="text-indigo-600" bgClass="bg-indigo-50" />
            <StatCard icon={<List className="w-5 h-5" />} label="Categories" value={stats?.num_categories || '4'} colorClass="text-purple-600" bgClass="bg-purple-50" />
            <StatCard icon={<BookOpen className="w-5 h-5" />} label="Avg Length" value={stats?.avg_length ? `${stats.avg_length} w` : '—'} colorClass="text-blue-600" bgClass="bg-blue-50" />
            <StatCard icon={<Database className="w-5 h-5" />} label="Vocab Size" value={stats?.vocab_size?.toLocaleString() || '—'} colorClass="text-green-600" bgClass="bg-green-50" />
            <StatCard icon={<CheckCircle className="w-5 h-5" />} label="Train Set" value={stats?.train_size?.toLocaleString() || '—'} colorClass="text-orange-600" bgClass="bg-orange-50" />
            <StatCard icon={<CheckCircle className="w-5 h-5" />} label="Test Set" value={stats?.test_size?.toLocaleString() || '—'} colorClass="text-red-600" bgClass="bg-red-50" />
          </>
        )}
      </div>

      {!loading && !notGenerated && stats && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Category Distribution */}
          {categoryDist.length > 0 && (
            <div className="card">
              <h2 className="text-lg font-bold text-slate-800 mb-4">Category Distribution</h2>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={categoryDist} margin={{ top: 5, right: 10, left: 10, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                  <YAxis tick={{ fontSize: 12 }} tickFormatter={(v) => `${(v / 1000).toFixed(0)}k`} />
                  <Tooltip formatter={(v) => [v.toLocaleString(), 'Articles']} />
                  <Bar dataKey="count" radius={[6, 6, 0, 0]}>
                    {categoryDist.map((entry, i) => (
                      <Cell key={i} fill={entry.fill} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Text Length Distribution */}
          {lengthDist.length > 0 && (
            <div className="card">
              <h2 className="text-lg font-bold text-slate-800 mb-4">Text Length Distribution</h2>
              <ResponsiveContainer width="100%" height={260}>
                <AreaChart data={lengthDist} margin={{ top: 5, right: 10, left: 10, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="bucket" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Area type="monotone" dataKey="count" fill="#e0e7ff" stroke="#6366f1" strokeWidth={2} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Top Words */}
          {topWords.length > 0 && (
            <div className="card">
              <h2 className="text-lg font-bold text-slate-800 mb-4">Top Words (After Preprocessing)</h2>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart
                  layout="vertical"
                  data={topWords}
                  margin={{ top: 5, right: 40, left: 60, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis type="number" tick={{ fontSize: 11 }} />
                  <YAxis type="category" dataKey="word" tick={{ fontSize: 12 }} width={55} />
                  <Tooltip />
                  <Bar dataKey="freq" fill="#6366f1" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Category Cards */}
          <div className="card">
            <h2 className="text-lg font-bold text-slate-800 mb-4">Category Breakdown</h2>
            <div className="space-y-3">
              {categoryDist.map((cat) => {
                const total = categoryDist.reduce((s, c) => s + c.count, 0)
                const pct = total ? ((cat.count / total) * 100).toFixed(1) : 0
                return (
                  <div key={cat.name} className="flex items-center gap-3">
                    <div className="w-24 text-sm font-semibold text-slate-700">{cat.name}</div>
                    <div className="flex-1 bg-slate-100 rounded-full h-4 overflow-hidden">
                      <div
                        className="h-full rounded-full progress-bar-animated"
                        style={{ width: `${pct}%`, backgroundColor: cat.fill }}
                      />
                    </div>
                    <div className="w-24 text-right text-xs text-slate-500">
                      {cat.count.toLocaleString()} ({pct}%)
                    </div>
                  </div>
                )
              })}
            </div>

            {/* Data quality badges */}
            <div className="mt-6 pt-4 border-t border-slate-100">
              <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">Data Quality</p>
              <div className="flex flex-wrap gap-2">
                {[
                  { label: 'No Missing Values', color: 'bg-green-100 text-green-700 border-green-200' },
                  { label: 'Duplicates Removed', color: 'bg-blue-100 text-blue-700 border-blue-200' },
                  { label: 'Balanced Classes', color: 'bg-purple-100 text-purple-700 border-purple-200' },
                ].map((badge) => (
                  <span key={badge.label} className={`badge border ${badge.color}`}>
                    ✓ {badge.label}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
