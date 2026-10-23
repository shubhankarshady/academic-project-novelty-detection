import React from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Cell,
  ReferenceLine,
} from 'recharts';
import { Card } from '../ui/Card';
import { getSectionLabel } from '../../utils/formatters';
import { Info } from 'lucide-react';

export function SectionCharts({ topMatch }) {
  if (!topMatch) return null;

  // Prepare data for Section Similarity Scores
  const simScores = topMatch.section_similarity_scores || {};
  const similarityChartData = Object.entries(simScores).map(([secKey, score]) => ({
    sectionKey: secKey,
    name: getSectionLabel(secKey),
    score: typeof score === 'number' ? Number(score.toFixed(3)) : 0,
  }));

  // Prepare data for Section Signed Contributions (Additive Ridge Regression)
  const contributions = topMatch.section_signed_contributions || {};
  const contributionChartData = Object.entries(contributions).map(([secKey, val]) => ({
    sectionKey: secKey,
    name: getSectionLabel(secKey),
    contribution: typeof val === 'number' ? Number(val.toFixed(4)) : 0,
  }));

  const customTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const data = payload[0];
      return (
        <div className="bg-slate-900 text-white text-xs p-2.5 rounded-lg shadow-lg border border-slate-800">
          <p className="font-semibold text-slate-200">{label}</p>
          <p className="text-indigo-400 mt-0.5">
            {data.name}: <span className="font-mono font-bold text-white">{data.value}</span>
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* 1. Section Similarity Scores Bar Chart */}
      <Card
        title="Section-Level Similarity Scores"
        subtitle={`Cosine similarity between candidate proposal and closest match (${topMatch.historical_project_id})`}
      >
        <div className="h-72 w-full pt-2">
          {similarityChartData.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={similarityChartData}
                margin={{ top: 10, right: 10, left: -15, bottom: 45 }}
              >
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                <XAxis
                  dataKey="name"
                  angle={-35}
                  textAnchor="end"
                  interval={0}
                  tick={{ fontSize: 11, fill: '#64748b' }}
                />
                <YAxis
                  domain={[0, 1]}
                  tick={{ fontSize: 11, fill: '#64748b' }}
                  tickFormatter={(v) => v.toFixed(2)}
                />
                <Tooltip content={customTooltip} />
                <Bar dataKey="score" name="Similarity Score" radius={[4, 4, 0, 0]}>
                  {similarityChartData.map((entry) => (
                    <Cell
                      key={`cell-${entry.sectionKey}`}
                      fill={entry.score >= 0.3 ? '#6366f1' : '#94a3b8'}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex items-center justify-center h-full text-slate-400 text-xs">
              No section similarity data available.
            </div>
          )}
        </div>
        <div className="mt-3 flex items-center justify-between text-[11px] text-slate-500 border-t border-slate-100 pt-3">
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-xs bg-indigo-500 inline-block" /> High overlap (≥ 0.30)
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-xs bg-slate-400 inline-block" /> Low / Differentiating (&lt; 0.30)
          </span>
        </div>
      </Card>

      {/* 2. Additive Ridge Signed Contributions Horizontal Bar Chart */}
      <Card
        title="Ridge Model Feature Contributions"
        subtitle="Individual section contribution values to the predicted similarity score"
      >
        <div className="h-72 w-full pt-2">
          {contributionChartData.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                layout="vertical"
                data={contributionChartData}
                margin={{ top: 10, right: 20, left: 25, bottom: 10 }}
              >
                <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#f1f5f9" />
                <XAxis
                  type="number"
                  tick={{ fontSize: 11, fill: '#64748b' }}
                  tickFormatter={(v) => v.toFixed(3)}
                />
                <YAxis
                  type="category"
                  dataKey="name"
                  tick={{ fontSize: 11, fill: '#64748b' }}
                  width={90}
                />
                <Tooltip content={customTooltip} />
                <ReferenceLine x={0} stroke="#94a3b8" strokeWidth={1} />
                <Bar dataKey="contribution" name="Ridge Contribution" radius={[0, 4, 4, 0]}>
                  {contributionChartData.map((entry) => (
                    <Cell
                      key={`contrib-${entry.sectionKey}`}
                      fill={entry.contribution >= 0 ? '#8b5cf6' : '#0ea5e9'}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex items-center justify-center h-full text-slate-400 text-xs">
              No feature contribution data available.
            </div>
          )}
        </div>
        <div className="mt-3 flex items-start gap-1.5 text-[11px] text-slate-500 border-t border-slate-100 pt-3">
          <Info className="w-3.5 h-3.5 text-indigo-500 shrink-0 mt-0.5" />
          <span>
            Model contributions are learned linear Ridge associations, not causal proof. Values reflect
            the product of section cosine similarity and learned feature weight.
          </span>
        </div>
      </Card>
    </div>
  );
}
