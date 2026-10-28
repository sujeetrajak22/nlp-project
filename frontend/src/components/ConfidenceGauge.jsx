export default function ConfidenceGauge({ confidence }) {
  const pct = Math.round(confidence * 100)

  let colorClass, bgClass, label, ringColor
  if (pct >= 70) {
    colorClass = 'text-green-600'
    bgClass = 'bg-green-50 border-green-200'
    label = 'High Confidence'
    ringColor = '#22c55e'
  } else if (pct >= 50) {
    colorClass = 'text-yellow-600'
    bgClass = 'bg-yellow-50 border-yellow-200'
    label = 'Moderate Confidence'
    ringColor = '#eab308'
  } else {
    colorClass = 'text-red-600'
    bgClass = 'bg-red-50 border-red-200'
    label = 'Low Confidence'
    ringColor = '#ef4444'
  }

  // SVG arc gauge
  const radius = 54
  const circumference = 2 * Math.PI * radius
  const offset = circumference - (pct / 100) * circumference

  return (
    <div className={`flex flex-col items-center gap-2 p-5 rounded-xl border ${bgClass}`}>
      <div className="relative w-32 h-32">
        <svg className="w-full h-full -rotate-90" viewBox="0 0 128 128">
          {/* Track */}
          <circle
            cx="64"
            cy="64"
            r={radius}
            fill="none"
            stroke="#e2e8f0"
            strokeWidth="10"
          />
          {/* Progress */}
          <circle
            cx="64"
            cy="64"
            r={radius}
            fill="none"
            stroke={ringColor}
            strokeWidth="10"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            className="transition-all duration-1000 ease-out"
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className={`text-3xl font-bold ${colorClass}`}>{pct}%</span>
        </div>
      </div>
      <div className="text-center">
        <p className={`text-sm font-semibold ${colorClass}`}>{label}</p>
        <p className="text-xs text-slate-500 mt-0.5">Prediction confidence</p>
      </div>
    </div>
  )
}
