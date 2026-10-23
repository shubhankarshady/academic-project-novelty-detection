import React from 'react';
import { AlertCircle, CheckCircle2, Info, AlertTriangle } from 'lucide-react';

export function Alert({
  children,
  title,
  variant = 'info',
  className = '',
  icon: CustomIcon,
}) {
  const styles = {
    info: {
      container: 'bg-indigo-50/70 text-indigo-900 border-indigo-200/80',
      icon: <Info className="w-5 h-5 text-indigo-600 shrink-0 mt-0.5" />,
      title: 'text-indigo-900',
    },
    success: {
      container: 'bg-emerald-50/80 text-emerald-900 border-emerald-200/80',
      icon: <CheckCircle2 className="w-5 h-5 text-emerald-600 shrink-0 mt-0.5" />,
      title: 'text-emerald-900',
    },
    warning: {
      container: 'bg-amber-50/80 text-amber-900 border-amber-200/80',
      icon: <AlertTriangle className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />,
      title: 'text-amber-900',
    },
    danger: {
      container: 'bg-rose-50/80 text-rose-900 border-rose-200/80',
      icon: <AlertCircle className="w-5 h-5 text-rose-600 shrink-0 mt-0.5" />,
      title: 'text-rose-900',
    },
  };

  const current = styles[variant] || styles.info;

  return (
    <div
      role="alert"
      className={`flex items-start gap-3 p-4 rounded-xl border text-sm ${current.container} ${className}`}
    >
      {CustomIcon ? <CustomIcon className="w-5 h-5 shrink-0 mt-0.5" /> : current.icon}
      <div className="flex-1 min-w-0">
        {title && <h4 className={`font-semibold mb-1 ${current.title}`}>{title}</h4>}
        <div className="leading-relaxed break-words">{children}</div>
      </div>
    </div>
  );
}
