import React from 'react';
import { getTierBadgeProps } from '../../utils/formatters';

export function NoveltyGauge({ percentage, tier }) {
  const tierProps = getTierBadgeProps(tier);
  const clamped = Math.max(0, Math.min(100, percentage ?? 0));

  // Circular gauge calculations
  const strokeWidth = 10;
  const radius = 54;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (clamped / 100) * circumference;

  let strokeColor = '#10b981'; // emerald
  if (clamped < 60) strokeColor = '#f43f5e'; // rose
  else if (clamped < 75) strokeColor = '#f59e0b'; // amber

  return (
    <div className="flex flex-col sm:flex-row items-center gap-6 p-6 bg-slate-900 text-white rounded-2xl shadow-md border border-slate-800">
      {/* Radial gauge SVG */}
      <div className="relative w-36 h-36 flex items-center justify-center shrink-0">
        <svg className="w-full h-full -rotate-90 transform" viewBox="0 0 130 130">
          {/* Background Track */}
          <circle
            cx="65"
            cy="65"
            r={radius}
            className="text-slate-800"
            strokeWidth={strokeWidth}
            stroke="currentColor"
            fill="transparent"
          />
          {/* Progress Arc */}
          <circle
            cx="65"
            cy="65"
            r={radius}
            stroke={strokeColor}
            strokeWidth={strokeWidth}
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            strokeLinecap="round"
            fill="transparent"
            className="transition-all duration-1000 ease-out"
          />
        </svg>

        <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
          <span className="text-3xl font-black tracking-tight text-white">
            {clamped.toFixed(1)}%
          </span>
          <span className="text-[10px] font-semibold tracking-wider text-slate-400 uppercase mt-0.5">
            Novelty
          </span>
        </div>
      </div>

      {/* Metric Breakdown */}
      <div className="space-y-2 text-center sm:text-left flex-1">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold border bg-slate-800 border-slate-700">
          <span className={`w-2 h-2 rounded-full ${tierProps.dotClass}`} />
          <span>{tier || 'Estimated Novelty'}</span>
        </div>

        <h3 className="text-xl font-bold text-white tracking-tight">
          Corpus-Relative Novelty Score
        </h3>

        <p className="text-xs text-slate-300 leading-relaxed max-w-xl">
          Estimated relative to 600 peer projects in the baseline reference corpus. A score of{' '}
          <strong className="text-white">{clamped.toFixed(1)}%</strong> indicates that the
          maximum learned multi-section similarity to any existing historical work is{' '}
          <strong className="text-white font-mono">{((100 - clamped) / 100).toFixed(3)}</strong>.
        </p>
      </div>
    </div>
  );
}
