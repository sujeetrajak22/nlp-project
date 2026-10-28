export default function LoadingSpinner({ size = 'md', color = 'indigo', label = '' }) {
  const sizeMap = {
    sm: 'w-4 h-4 border-2',
    md: 'w-6 h-6 border-2',
    lg: 'w-10 h-10 border-3',
    xl: 'w-14 h-14 border-4',
  }
  const colorMap = {
    indigo: 'border-indigo-600 border-t-transparent',
    white: 'border-white border-t-transparent',
    slate: 'border-slate-400 border-t-transparent',
  }

  return (
    <div className="flex flex-col items-center justify-center gap-2">
      <div
        className={`${sizeMap[size] || sizeMap.md} ${
          colorMap[color] || colorMap.indigo
        } rounded-full animate-spin`}
        role="status"
        aria-label={label || 'Loading'}
      />
      {label && (
        <span className="text-sm text-slate-500 animate-pulse">{label}</span>
      )}
    </div>
  )
}
