import React from 'react';
import { Menu, Database, RefreshCw, CheckCircle2, AlertCircle } from 'lucide-react';
import { Badge } from '../ui/Badge';

export function Header({
  title,
  subtitle,
  healthData,
  healthLoading,
  onRefreshHealth,
  onToggleMobileMenu,
}) {
  const isHealthy = healthData?.status === 'healthy';

  return (
    <header className="sticky top-0 z-30 bg-white/90 backdrop-blur-md border-b border-slate-200/80 px-4 sm:px-8 py-3.5 flex items-center justify-between gap-4 transition-all">
      <div className="flex items-center gap-3 min-w-0">
        <button
          type="button"
          onClick={onToggleMobileMenu}
          className="md:hidden p-2 rounded-lg text-slate-500 hover:text-slate-800 hover:bg-slate-100 cursor-pointer"
          aria-label="Open navigation menu"
        >
          <Menu className="w-5 h-5" />
        </button>

        <div className="min-w-0">
          <h2 className="text-lg sm:text-xl font-bold text-slate-900 tracking-tight truncate">
            {title}
          </h2>
          {subtitle && (
            <p className="text-xs sm:text-sm text-slate-500 truncate mt-0.5">{subtitle}</p>
          )}
        </div>
      </div>

      <div className="flex items-center gap-2 sm:gap-3 shrink-0">
        {/* Service Status Badge */}
        {healthLoading ? (
          <Badge variant="neutral" dot size="sm">
            Checking backend...
          </Badge>
        ) : isHealthy ? (
          <div className="flex items-center gap-2">
            <span className="hidden sm:inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-emerald-50 text-emerald-700 border border-emerald-200/80">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
              API Connected
            </span>
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-indigo-50 text-indigo-700 border border-indigo-200/80">
              <Database className="w-3.5 h-3.5 text-indigo-600" />
              <span className="font-semibold">{healthData?.reference_corpus_size ?? 600}</span>
              <span className="hidden md:inline">Ref Projects</span>
            </span>
          </div>
        ) : (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-rose-50 text-rose-700 border border-rose-200/80">
            <AlertCircle className="w-3.5 h-3.5 text-rose-500" />
            Backend Disconnected
          </span>
        )}

        {/* Refresh health button */}
        <button
          type="button"
          onClick={onRefreshHealth}
          disabled={healthLoading}
          className="p-1.5 text-slate-400 hover:text-slate-600 hover:bg-slate-100 rounded-lg transition-colors cursor-pointer disabled:opacity-50"
          title="Refresh server status"
          aria-label="Refresh server status"
        >
          <RefreshCw className={`w-4 h-4 ${healthLoading ? 'animate-spin' : ''}`} />
        </button>
      </div>
    </header>
  );
}
