export default function StatCard({ icon, label, value, trend, colorClass = 'text-indigo-600', bgClass = 'bg-indigo-50' }) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5 flex items-start gap-4 hover:shadow-md transition-shadow">
      {icon && (
        <div className={`w-10 h-10 ${bgClass} rounded-lg flex items-center justify-center flex-shrink-0`}>
          <span className={`text-lg ${colorClass}`}>{icon}</span>
        </div>
      )}
      <div className="flex-1 min-w-0">
        <p className="text-xs text-slate-500 uppercase tracking-wider font-medium mb-0.5">{label}</p>
        <p className={`text-2xl font-bold ${colorClass} truncate`}>{value}</p>
        {trend && (
          <p className="text-xs text-slate-400 mt-0.5">{trend}</p>
        )}
      </div>
    </div>
  )
}
