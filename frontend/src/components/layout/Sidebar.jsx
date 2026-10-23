import React from 'react';
import {
  LayoutDashboard,
  Sparkles,
  History,
  GraduationCap,
  Database,
  ExternalLink,
} from 'lucide-react';

export function Sidebar({ currentTab, setCurrentTab, healthData, isMobileOpen, setIsMobileOpen }) {
  const navItems = [
    {
      id: 'dashboard',
      label: 'Dashboard',
      icon: LayoutDashboard,
      description: 'System overview & metrics',
    },
    {
      id: 'evaluate',
      label: 'Evaluate Proposal',
      icon: Sparkles,
      description: 'Analyze novelty & XAI report',
    },
    {
      id: 'history',
      label: 'Proposal History',
      icon: History,
      description: 'Saved proposal evaluations',
    },
  ];

  const handleSelect = (id) => {
    setCurrentTab(id);
    if (setIsMobileOpen) setIsMobileOpen(false);
  };

  return (
    <>
      {/* Mobile backdrop */}
      {isMobileOpen && (
        <div
          className="fixed inset-0 z-40 bg-slate-900/50 md:hidden"
          onClick={() => setIsMobileOpen(false)}
        />
      )}

      {/* Sidebar container */}
      <aside
        className={`fixed md:sticky top-0 left-0 z-40 h-screen w-64 bg-white border-r border-slate-200 flex flex-col justify-between transition-transform duration-200 ease-in-out md:translate-x-0 ${
          isMobileOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        <div>
          {/* Brand header */}
          <div className="h-16 px-6 border-b border-slate-100 flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-indigo-600 to-purple-600 flex items-center justify-center text-white shadow-xs">
              <GraduationCap className="w-5 h-5" />
            </div>
            <div>
              <h1 className="text-sm font-bold tracking-tight text-slate-900 leading-tight">
                Academic Novelty
              </h1>
              <p className="text-[11px] font-medium text-indigo-600 uppercase tracking-wider">
                Detection System
              </p>
            </div>
          </div>

          {/* Navigation Links */}
          <nav className="p-4 space-y-1.5" aria-label="Sidebar navigation">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = currentTab === item.id;
              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => handleSelect(item.id)}
                  className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-left transition-all cursor-pointer font-medium text-sm group ${
                    isActive
                      ? 'bg-indigo-50 text-indigo-700 font-semibold shadow-2xs border border-indigo-100/80'
                      : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
                  }`}
                >
                  <Icon
                    className={`w-4 h-4 shrink-0 transition-colors ${
                      isActive ? 'text-indigo-600' : 'text-slate-400 group-hover:text-slate-600'
                    }`}
                  />
                  <div className="flex-1 truncate">
                    <div>{item.label}</div>
                  </div>
                  {isActive && (
                    <div className="w-1.5 h-1.5 rounded-full bg-indigo-600 shrink-0" />
                  )}
                </button>
              );
            })}
          </nav>
        </div>

        {/* Bottom reference corpus indicator */}
        <div className="p-4 border-t border-slate-100 bg-slate-50/60">
          <div className="rounded-xl p-3 bg-white border border-slate-200/80 shadow-2xs space-y-2">
            <div className="flex items-center justify-between text-xs font-semibold text-slate-800">
              <span className="flex items-center gap-1.5 text-slate-700">
                <Database className="w-3.5 h-3.5 text-indigo-600" />
                Reference Corpus
              </span>
              <span className="inline-flex items-center px-1.5 py-0.5 rounded-md text-[10px] font-medium bg-emerald-50 text-emerald-700 border border-emerald-200">
                Active
              </span>
            </div>
            <div className="text-[11px] text-slate-500 leading-normal">
              Comparing against{' '}
              <strong className="text-slate-700">
                {healthData?.reference_corpus_size ?? 600}
              </strong>{' '}
              baseline historical research projects.
            </div>
          </div>
        </div>
      </aside>
    </>
  );
}
