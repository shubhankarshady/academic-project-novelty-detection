import React, { useState } from 'react';
import {
  ChevronDown,
  ChevronUp,
  FileCheck,
  Tag,
  AlertTriangle,
  Sparkles,
  Layers,
  Quote,
} from 'lucide-react';
import { Badge } from '../ui/Badge';
import { formatPercent, formatScore, getSectionLabel } from '../../utils/formatters';

export function SimilarProjectCard({ match, rank, isTopMatch = false }) {
  const [isExpanded, setIsExpanded] = useState(isTopMatch);
  const [selectedSectionEvidence, setSelectedSectionEvidence] = useState(null);

  const diffSummary = match.difference_summary || {
    high_overlap_sections: [],
    differentiating_sections: [],
  };

  const sectionEvidence = match.section_evidence || {};
  const rankedSections = match.ranked_sections || [];

  return (
    <div
      className={`rounded-xl border transition-all ${
        isTopMatch
          ? 'border-indigo-300 bg-white shadow-sm ring-1 ring-indigo-200'
          : 'border-slate-200 bg-white hover:border-slate-300'
      }`}
    >
      {/* Card Header */}
      <div
        onClick={() => setIsExpanded((prev) => !prev)}
        className="p-5 flex flex-col md:flex-row items-start md:items-center justify-between gap-4 cursor-pointer select-none"
      >
        <div className="flex items-start gap-3.5 min-w-0">
          <div
            className={`w-8 h-8 rounded-lg flex items-center justify-center font-bold text-xs shrink-0 ${
              isTopMatch
                ? 'bg-indigo-600 text-white'
                : 'bg-slate-100 text-slate-700'
            }`}
          >
            #{rank}
          </div>

          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2 mb-1">
              <span className="font-mono text-xs font-bold text-indigo-700 bg-indigo-50 px-2 py-0.5 rounded border border-indigo-100">
                {match.historical_project_id}
              </span>
              <span className="text-xs text-slate-500 font-medium">
                {match.historical_domain || 'General'}
              </span>
              {match.scoring_mode === 'hybrid' ? (
                <Badge variant="purple" size="sm">
                  Hybrid (α={match.hybrid_alpha ?? 0.5})
                </Badge>
              ) : (
                <Badge variant="neutral" size="sm">
                  TF-IDF Lexical
                </Badge>
              )}
              {isTopMatch && (
                <Badge variant="indigo" size="sm">
                  Closest Historical Match
                </Badge>
              )}
            </div>

            <h4 className="text-base font-semibold text-slate-900 leading-snug">
              {match.historical_project_title}
            </h4>
          </div>
        </div>

        {/* Right Metric Pill & Toggle */}
        <div className="flex items-center gap-4 shrink-0 self-end md:self-auto">
          <div className="text-right">
            <div className="text-xs text-slate-500 font-medium">Estimated Novelty</div>
            <div className="text-lg font-black text-slate-900 font-mono">
              {formatPercent(match.novelty_score_pct)}
            </div>
            <div className="text-[11px] text-slate-500 font-mono">
              Similarity: {formatScore(match.predicted_similarity)} / 4.0
            </div>
            <div className="text-[11px] font-semibold text-indigo-600 mt-0.5">
              {match.novelty_category?.includes('Duplicate')
                ? 'Requires Faculty Review'
                : match.novelty_category || 'Novelty Assessment'}
            </div>
          </div>

          <button
            type="button"
            className="p-2 text-slate-400 hover:text-slate-600 rounded-lg hover:bg-slate-100 transition-colors"
            aria-label={isExpanded ? 'Collapse match details' : 'Expand match details'}
          >
            {isExpanded ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
          </button>
        </div>
      </div>

      {/* Expanded Accordion Body */}
      {isExpanded && (
        <div className="px-5 pb-5 pt-2 border-t border-slate-100 space-y-5 text-sm text-slate-700">
          {/* Overlap & Differentiation Summary */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-3.5 rounded-lg bg-amber-50/70 border border-amber-200/70 text-xs">
              <div className="font-semibold text-amber-900 flex items-center gap-1.5 mb-1">
                <AlertTriangle className="w-4 h-4 text-amber-600" />
                High Overlap Sections (Similarity ≥ 0.30)
              </div>
              {diffSummary.high_overlap_sections && diffSummary.high_overlap_sections.length > 0 ? (
                <div className="flex flex-wrap gap-1.5 mt-2">
                  {diffSummary.high_overlap_sections.map((sec) => (
                    <span
                      key={sec}
                      className="px-2 py-0.5 rounded-md bg-amber-100 text-amber-800 font-medium"
                    >
                      {getSectionLabel(sec)}
                    </span>
                  ))}
                </div>
              ) : (
                <p className="text-amber-700 mt-1">No sections exceed the high similarity threshold.</p>
              )}
            </div>

            <div className="p-3.5 rounded-lg bg-emerald-50/70 border border-emerald-200/70 text-xs">
              <div className="font-semibold text-emerald-900 flex items-center gap-1.5 mb-1">
                <Sparkles className="w-4 h-4 text-emerald-600" />
                Differentiating Sections (Similarity &lt; 0.15)
              </div>
              {diffSummary.differentiating_sections && diffSummary.differentiating_sections.length > 0 ? (
                <div className="flex flex-wrap gap-1.5 mt-2">
                  {diffSummary.differentiating_sections.map((sec) => (
                    <span
                      key={sec}
                      className="px-2 py-0.5 rounded-md bg-emerald-100 text-emerald-800 font-medium"
                    >
                      {getSectionLabel(sec)}
                    </span>
                  ))}
                </div>
              ) : (
                <p className="text-emerald-700 mt-1">
                  Candidate shares baseline methodology across most sections.
                </p>
              )}
            </div>
          </div>

          {/* Ranked Sections Table */}
          {rankedSections.length > 0 && (() => {
            const isHybrid =
              match.scoring_mode === 'hybrid' ||
              rankedSections.some(
                (r) => r.hybrid_similarity !== undefined && r.hybrid_similarity !== null
              );

            return (
              <div>
                <div className="flex items-center justify-between mb-2">
                  <h5 className="text-xs font-bold uppercase tracking-wider text-slate-500 flex items-center gap-1.5">
                    <Layers className="w-3.5 h-3.5 text-indigo-500" />
                    Section-by-Section Feature Breakdown
                  </h5>
                  <span className="text-[11px] text-slate-500 font-medium">
                    {isHybrid
                      ? `Hybrid Mode (α=${match.hybrid_alpha ?? 0.50}: ${((match.hybrid_alpha ?? 0.5) * 100).toFixed(0)}% Lexical / ${(100 - (match.hybrid_alpha ?? 0.5) * 100).toFixed(0)}% Semantic)`
                      : 'Production Mode: Exact Lexical TF-IDF'}
                  </span>
                </div>
                <div className="overflow-x-auto rounded-lg border border-slate-200">
                  <table className="w-full text-left text-xs">
                    <thead className="bg-slate-50 text-slate-600 font-semibold border-b border-slate-200">
                      <tr>
                        <th className="py-2.5 px-3">Section</th>
                        {isHybrid ? (
                          <>
                            <th className="py-2.5 px-3">Lexical (TF-IDF)</th>
                            <th className="py-2.5 px-3">Semantic (Dense)</th>
                            <th className="py-2.5 px-3">Combined (Hybrid)</th>
                          </>
                        ) : (
                          <>
                            <th className="py-2.5 px-3">Cosine Similarity</th>
                            <th className="py-2.5 px-3 text-slate-400 font-normal">
                              Semantic (Dense)
                            </th>
                          </>
                        )}
                        <th className="py-2.5 px-3">Model Weight</th>
                        <th className="py-2.5 px-3">Contribution</th>
                        <th className="py-2.5 px-3 text-right">Shared Terms</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100 font-normal">
                      {rankedSections.map((row) => {
                        const evidence = sectionEvidence[row.section];
                        const sharedCount = evidence?.shared_terms_count ?? 0;
                        const isSelected = selectedSectionEvidence === row.section;

                        return (
                          <tr
                            key={row.section}
                            onClick={() =>
                              setSelectedSectionEvidence(isSelected ? null : row.section)
                            }
                            className={`hover:bg-slate-50 cursor-pointer transition-colors ${
                              isSelected ? 'bg-indigo-50/60 font-medium' : ''
                            }`}
                          >
                            <td className="py-2.5 px-3 font-medium text-slate-900">
                              {getSectionLabel(row.section)}
                            </td>
                            {isHybrid ? (
                              <>
                                <td className="py-2.5 px-3 font-mono text-slate-700">
                                  {formatScore(row.lexical_similarity ?? row.raw_similarity)}
                                </td>
                                <td className="py-2.5 px-3 font-mono text-purple-700 font-medium">
                                  {row.semantic_similarity !== undefined &&
                                  row.semantic_similarity !== null
                                    ? formatScore(row.semantic_similarity)
                                    : '—'}
                                </td>
                                <td className="py-2.5 px-3 font-mono font-bold text-indigo-700">
                                  {formatScore(row.hybrid_similarity ?? row.raw_similarity)}
                                </td>
                              </>
                            ) : (
                              <>
                                <td className="py-2.5 px-3 font-mono">
                                  {formatScore(row.raw_similarity)}
                                </td>
                                <td className="py-2.5 px-3 text-[11px] text-slate-400 italic">
                                  Unavailable in TF-IDF mode
                                </td>
                              </>
                            )}
                            <td className="py-2.5 px-3 font-mono text-slate-500">
                              {row.model_weight.toFixed(4)}
                            </td>
                            <td className="py-2.5 px-3 font-mono">
                              <span
                                className={
                                  row.signed_contribution >= 0
                                    ? 'text-indigo-600 font-semibold'
                                    : 'text-sky-600'
                                }
                              >
                                {row.signed_contribution > 0 ? '+' : ''}
                                {row.signed_contribution.toFixed(4)}
                              </span>
                            </td>
                            <td className="py-2.5 px-3 text-right">
                              <span className="inline-flex items-center px-2 py-0.5 rounded text-[11px] bg-slate-100 text-slate-700">
                                {sharedCount} terms
                              </span>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            );
          })()}

          {/* Section Evidence Snippets (Candidate vs Historical) */}
          <div>
            <h5 className="text-xs font-bold uppercase tracking-wider text-slate-500 mb-2 flex items-center gap-1.5">
              <Quote className="w-3.5 h-3.5 text-indigo-500" />
              Source-Verified Snippets & Shared Informative Terms
            </h5>

            {/* Section tabs for evidence */}
            <div className="flex flex-wrap gap-1.5 mb-3">
              {Object.keys(sectionEvidence).map((secKey) => {
                const isActive = (selectedSectionEvidence || 'abstract') === secKey;
                return (
                  <button
                    key={secKey}
                    type="button"
                    onClick={() => setSelectedSectionEvidence(secKey)}
                    className={`px-2.5 py-1 text-xs rounded-lg transition-all cursor-pointer font-medium ${
                      isActive
                        ? 'bg-indigo-600 text-white shadow-2xs'
                        : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                    }`}
                  >
                    {getSectionLabel(secKey)}
                  </button>
                );
              })}
            </div>

            {/* Selected Section Evidence View */}
            {(() => {
              const activeSec = selectedSectionEvidence || 'abstract';
              const ev = sectionEvidence[activeSec];
              if (!ev || !ev.shared_terms || ev.shared_terms.length === 0) {
                return (
                  <div className="p-4 rounded-lg bg-slate-50 border border-slate-200 text-xs text-slate-500">
                    No shared technical keywords detected in the{' '}
                    <strong className="text-slate-700">{getSectionLabel(activeSec)}</strong>{' '}
                    section above the TF-IDF threshold.
                  </div>
                );
              }

              return (
                <div className="space-y-3">
                  {ev.shared_terms.map((termItem, idx) => (
                    <div
                      key={idx}
                      className="p-3.5 rounded-lg border border-slate-200 bg-slate-50/50 space-y-2 text-xs"
                    >
                      <div className="flex items-center justify-between gap-2 border-b border-slate-200/60 pb-2">
                        <span className="font-semibold text-slate-900 flex items-center gap-1.5">
                          <Tag className="w-3.5 h-3.5 text-indigo-600" />
                          Shared Technical Term: &ldquo;{termItem.term}&rdquo;
                        </span>
                        <span className="font-mono text-[11px] text-slate-500">
                          Cosine contrib: {termItem.term_cosine_contribution.toFixed(4)}
                        </span>
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-1">
                        <div className="p-2.5 rounded bg-white border border-slate-200">
                          <div className="text-[10px] font-bold uppercase tracking-wider text-indigo-700 mb-1">
                            Candidate Proposal Context
                          </div>
                          <p className="text-slate-700 italic leading-relaxed">
                            &ldquo;{termItem.candidate_snippet}&rdquo;
                          </p>
                        </div>

                        <div className="p-2.5 rounded bg-white border border-slate-200">
                          <div className="text-[10px] font-bold uppercase tracking-wider text-purple-700 mb-1">
                            Historical Project Context ({match.historical_project_id})
                          </div>
                          <p className="text-slate-700 italic leading-relaxed">
                            &ldquo;{termItem.historical_snippet}&rdquo;
                          </p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              );
            })()}
          </div>
        </div>
      )}
    </div>
  );
}
