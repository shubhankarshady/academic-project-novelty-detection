import React, { useState, useEffect } from 'react';
import {
  FileText,
  Calendar,
  Layers,
  Sparkles,
  ExternalLink,
  ShieldCheck,
  Tag,
  AlertCircle,
} from 'lucide-react';
import { Modal } from '../ui/Modal';
import { Badge } from '../ui/Badge';
import { LoadingSpinner } from '../ui/LoadingSpinner';
import { Alert } from '../ui/Alert';
import { SectionCharts } from '../report/SectionCharts';
import { SimilarProjectCard } from '../report/SimilarProjectCard';
import {
  formatPercent,
  formatScore,
  formatDate,
  getTierBadgeProps,
  getSectionLabel,
  SECTION_CONFIG,
} from '../../utils/formatters';
import { getProposalById } from '../../services/api';

export function ProposalDetailModal({ projectId, isOpen, onClose }) {
  const [proposal, setProposal] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('sections'); // 'sections' | 'report'

  useEffect(() => {
    if (isOpen && projectId) {
      loadProposalDetails();
    } else {
      setProposal(null);
      setError(null);
    }
  }, [isOpen, projectId]);

  const loadProposalDetails = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getProposalById(projectId);
      setProposal(data);
    } catch (err) {
      setError(err.message || 'Failed to fetch proposal details.');
    } finally {
      setLoading(false);
    }
  };

  const report = proposal?.evaluation_report;
  const tierProps = getTierBadgeProps(proposal?.novelty_assessment_tier);

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title={proposal ? `Proposal: ${proposal.project_title}` : 'Proposal Details'}
      maxWidth="max-w-5xl"
    >
      {loading ? (
        <LoadingSpinner text="Retrieving stored proposal and evaluation report..." />
      ) : error ? (
        <div className="space-y-4">
          <Alert variant="danger" title="Error Loading Proposal">
            {error}
          </Alert>
          <div className="text-right">
            <button
              onClick={loadProposalDetails}
              className="text-xs text-indigo-600 hover:underline font-semibold"
            >
              Try Again
            </button>
          </div>
        </div>
      ) : proposal ? (
        <div className="space-y-6">
          {/* Header Summary Banner */}
          <div className="p-4 rounded-xl bg-slate-900 text-white flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
            <div>
              <div className="flex items-center gap-2 mb-1">
                <span className="font-mono text-xs text-indigo-300 font-semibold bg-indigo-950/80 px-2 py-0.5 rounded border border-indigo-800">
                  {proposal.project_id}
                </span>
                <span className="text-xs text-slate-400">
                  {proposal.domain || 'General'}
                </span>
              </div>
              <h4 className="text-base font-bold text-white leading-snug">
                {proposal.project_title}
              </h4>
              <p className="text-xs text-slate-400 mt-1 flex items-center gap-1.5">
                <Calendar className="w-3.5 h-3.5" />
                Submitted: {formatDate(proposal.created_at || proposal.submission_timestamp)}
              </p>
            </div>

            <div className="flex items-center gap-4 bg-slate-800/80 p-3 rounded-lg border border-slate-700">
              <div className="text-center">
                <div className="text-[10px] uppercase font-bold text-slate-400">Novelty</div>
                <div className="text-lg font-black text-white">
                  {formatPercent(proposal.novelty_score_pct)}
                </div>
              </div>
              <div className="h-8 w-px bg-slate-700" />
              <div className="text-center">
                <div className="text-[10px] uppercase font-bold text-slate-400">Max Similarity</div>
                <div className="text-lg font-black font-mono text-white">
                  {formatScore(proposal.max_similarity_score)}
                </div>
              </div>
            </div>
          </div>

          {/* Navigation Tabs */}
          <div className="flex border-b border-slate-200 gap-6">
            <button
              type="button"
              onClick={() => setActiveTab('sections')}
              className={`pb-3 text-sm font-semibold cursor-pointer border-b-2 transition-colors ${
                activeTab === 'sections'
                  ? 'border-indigo-600 text-indigo-600'
                  : 'border-transparent text-slate-500 hover:text-slate-800'
              }`}
            >
              8 Proposal Sections
            </button>
            <button
              type="button"
              onClick={() => setActiveTab('report')}
              className={`pb-3 text-sm font-semibold cursor-pointer border-b-2 transition-colors ${
                activeTab === 'report'
                  ? 'border-indigo-600 text-indigo-600'
                  : 'border-transparent text-slate-500 hover:text-slate-800'
              }`}
            >
              Full Evaluation & XAI Report
            </button>
          </div>

          {/* Tab 1: Proposal Sections */}
          {activeTab === 'sections' && (
            <div className="space-y-4">
              {SECTION_CONFIG.map((sec, idx) => {
                const text = proposal[sec.key];
                if (!text && sec.key === 'domain') return null;

                return (
                  <div
                    key={sec.key}
                    className="p-4 rounded-xl border border-slate-200 bg-slate-50/40 space-y-1.5"
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-bold uppercase tracking-wider text-slate-500 flex items-center gap-1.5">
                        <span className="w-4 h-4 rounded bg-slate-200 text-slate-700 text-[10px] flex items-center justify-center font-bold">
                          {idx + 1}
                        </span>
                        {sec.label}
                      </span>
                      <span className="text-[11px] text-slate-400 font-mono">
                        {(text || '').length} chars
                      </span>
                    </div>
                    <p className="text-sm text-slate-800 whitespace-pre-wrap leading-relaxed">
                      {text || 'Not provided'}
                    </p>
                  </div>
                );
              })}
            </div>
          )}

          {/* Tab 2: Full Saved Evaluation Report */}
          {activeTab === 'report' && (
            <div className="space-y-6">
              {report ? (
                <>
                  <div className="p-4 rounded-xl bg-amber-50 border border-amber-200 text-xs text-amber-900">
                    <strong>Corpus Note:</strong> Saved evaluations capture the novelty state at the time
                    of scoring relative to the 600 baseline reference projects. User submissions are stored
                    separately and do not contaminate the baseline reference corpus.
                  </div>

                  {/* Section Visualizations if top matches available */}
                  {report.top_matches && report.top_matches.length > 0 && (
                    <SectionCharts topMatch={report.top_matches[0]} />
                  )}

                  {/* Top Historical Matches */}
                  {report.top_matches && report.top_matches.length > 0 ? (
                    <div className="space-y-3">
                      <h5 className="text-sm font-bold text-slate-900">
                        Top Historical Matches at Evaluation Time
                      </h5>
                      {report.top_matches.map((match, idx) => (
                        <SimilarProjectCard
                          key={match.historical_project_id || idx}
                          match={match}
                          rank={idx + 1}
                          isTopMatch={idx === 0}
                        />
                      ))}
                    </div>
                  ) : (
                    <p className="text-xs text-slate-500">No match records saved in the report.</p>
                  )}
                </>
              ) : (
                <div className="p-6 text-center text-sm text-slate-500">
                  No detailed evaluation report JSON saved for this entry.
                </div>
              )}
            </div>
          )}
        </div>
      ) : null}
    </Modal>
  );
}
