import { getCategoryStyle } from './CategoryBadge'

export default function ProbabilityBar({ category, probability, isTop = false }) {
  const style = getCategoryStyle(category)
  const pct = Math.round(probability * 100)

  const barColorMap = {
    World: 'bg-blue-500',
    Sports: 'bg-green-500',
    Business: 'bg-orange-500',
    'Sci/Tech': 'bg-purple-500',
  }
  const barColor = barColorMap[category] || 'bg-indigo-500'

  return (
    <div className={`flex items-center gap-3 p-2 rounded-lg transition-colors ${isTop ? 'bg-slate-50' : ''}`}>
      <div className="w-20 text-right flex-shrink-0">
        <span className={`text-xs font-semibold ${style.text}`}>{category}</span>
      </div>
      <div className="flex-1 bg-slate-200 rounded-full h-3 overflow-hidden">
        <div
          className={`h-full rounded-full progress-bar-animated ${barColor} ${isTop ? 'opacity-100' : 'opacity-60'}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <div className="w-12 flex-shrink-0">
        <span className={`text-xs font-bold ${isTop ? style.text : 'text-slate-500'}`}>
          {pct}%
        </span>
      </div>
    </div>
  )
}
