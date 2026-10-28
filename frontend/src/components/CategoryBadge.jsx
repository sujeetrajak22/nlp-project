const CATEGORY_STYLES = {
  World: {
    bg: 'bg-blue-100',
    text: 'text-blue-700',
    border: 'border-blue-200',
    dot: 'bg-blue-500',
    emoji: '🌍',
  },
  Sports: {
    bg: 'bg-green-100',
    text: 'text-green-700',
    border: 'border-green-200',
    dot: 'bg-green-500',
    emoji: '⚽',
  },
  Business: {
    bg: 'bg-orange-100',
    text: 'text-orange-700',
    border: 'border-orange-200',
    dot: 'bg-orange-500',
    emoji: '💼',
  },
  'Sci/Tech': {
    bg: 'bg-purple-100',
    text: 'text-purple-700',
    border: 'border-purple-200',
    dot: 'bg-purple-500',
    emoji: '💻',
  },
}

export function getCategoryStyle(category) {
  return CATEGORY_STYLES[category] || {
    bg: 'bg-slate-100',
    text: 'text-slate-700',
    border: 'border-slate-200',
    dot: 'bg-slate-500',
    emoji: '📰',
  }
}

export default function CategoryBadge({ category, size = 'md', showEmoji = true }) {
  const style = getCategoryStyle(category)
  const sizeClasses = {
    sm: 'text-xs px-2 py-0.5',
    md: 'text-sm px-3 py-1',
    lg: 'text-base px-4 py-1.5 font-semibold',
    xl: 'text-xl px-5 py-2 font-bold',
  }

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border font-medium ${style.bg} ${style.text} ${style.border} ${sizeClasses[size] || sizeClasses.md}`}
    >
      {showEmoji && <span>{style.emoji}</span>}
      {category}
    </span>
  )
}
