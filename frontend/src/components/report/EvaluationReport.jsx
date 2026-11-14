import React, { useState } from 'react';
import {
  Save,
  CheckCircle2,
  AlertTriangle,
  ExternalLink,
  ShieldAlert,
  Layers,
  Database,
  ArrowRight,
} from 'lucide-react';
import { Card } from '../ui/Card';
import { Button } from '../ui/Button';
import { Alert } from '../ui/Alert';
import { Badge } from '../ui/Badge';
import { NoveltyGauge } from './NoveltyGauge';
import { SectionCharts } from './SectionCharts';
import { SimilarProjectCard } from './SimilarProjectCard';
import { AIExplanationPanel } from './AIExplanationPanel';
import { formatPercent, formatScore, formatDate, getTierBadgeProps } from '../../utils/formatters';

export function EvaluationReport({
  report,
  proposalData,
  onSave,
  saveLoading,
  saveSuccess,
  saveError,
  isDirtyAfterScoring,
  onViewInHistory,
  onReevaluate,
}) {
  if (!report) return null;

  const topMatches = report.top_matches || [];
  const topMatch = topMatches.length > 0 ? topMatches[0] : null;
  const tierProps = getTierBadgeProps(report.novelty_assessment_tier);

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* 1. Mandatory Academic Disclaimer Notice */}
      <div className="bg-amber-50/90 border border-amber-200/90 rounded-2xl p-5 shadow-2xs">
        <div className="flex items-start gap-3.5">
          <ShieldAlert className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <h4 className="text-xs font-bold uppercase tracking-wider text-amber-900">
              Corpus-Relative Novelty Assessment Notice
            </h4>
            <p className="text-sm font-medium text-amber-950 leading-relaxed">
              &ldquo;Novelty is estimated relative to the current reference corpus. This score does
              not prove originality, guarantee acceptance, or establish plagiarism.&rdquo;
            </p>
            <p className="text-xs text-amber-800 pt-1">
              Evaluated against {report.corpus_size_evaluated || 600} reference research initiatives.
              The novelty score is mathematically formulated as (1 - max similarity) × 100%.
            </p>
          </div>
        </div>
      </div>

      {/* 2. Top Summary & Gauge */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <NoveltyGauge
            percentage={report.overall_novelty_score_pct}
            tier={report.novelty_assessment_tier}
          />
        </div>

        {/* Core Metric Cards */}
        <div className="flex flex-col justify-between gap-3">
          <Card className="flex-1 flex flex-col justify-center">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
              Maximum Historical Similarity
            </span>
            <div className="mt-2 text-2xl font-black font-mono text-slate-900">
              {formatScore(report.max_similarity_score)}
            </div>
            <p className="text-xs text-slate-500 mt-1">
              Top match ID:{' '}
              <strong className="text-indigo-600 font-mono">
                {topMatch?.historical_project_id || 'N/A'}
              </strong>
            </p>
          </Card>

          <Card className="flex-1 flex flex-col justify-center bg-slate-50/70 border-slate-200">
            <div className="flex items-center justify-between gap-1 mb-1">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                Evaluation Model
              </span>
              {report.scoring_mode === 'tuned_hybrid' ? (
                <Badge variant="indigo" size="sm" dot>
                  Two-Tier Hybrid (α={report.hybrid_alpha ?? 0.60})
                </Badge>
              ) : report.scoring_mode === 'hybrid' ? (
                <Badge variant="purple" size="sm">
                  Hybrid Baseline (α={report.hybrid_alpha ?? 0.50})
                </Badge>
              ) : (
                <Badge variant="neutral" size="sm">
                  TF-IDF Baseline
                </Badge>
              )}
            </div>
            <div className="text-xs font-mono font-semibold text-slate-800">
              {report.scoring_mode === 'tuned_hybrid'
                ? `60% TF-IDF + 40% Dense Semantic (α = ${report.hybrid_alpha ?? 0.60})`
                : report.scoring_mode === 'hybrid'
                ? `50% TF-IDF + 50% Dense (α = ${report.hybrid_alpha ?? 0.50})`
                : '100% Lexical TF-IDF'}
            </div>
            {report.scoring_formula && (
              <div className="text-[10px] text-slate-500 font-mono mt-1 line-clamp-1" title={report.scoring_formula}>
                {report.scoring_formula}
              </div>
            )}
            {report.retrieval_stage === 'two_stage' && (
              <div className="text-[10px] text-indigo-700 font-medium bg-indigo-50 border border-indigo-200/60 rounded px-1.5 py-0.5 mt-1">
                Two-Tier Retrieval: Top {report.stage2_shortlist_size || 50} candidate shortlist from {report.stage1_candidates_evaluated || 600} projects
              </div>
            )}
            <div className="text-[10px] text-slate-400 mt-0.5">
              Version: {report.model_version || '1.0'}
            </div>
          </Card>

          <Card className="flex-1 flex flex-col justify-center bg-slate-50/70 border-slate-200">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
              Candidate Proposal Title
            </span>
            <div className="mt-1 text-sm font-bold text-slate-900 line-clamp-2">
              {report.candidate_proposal_title}
            </div>
            <div className="text-[11px] text-slate-400 mt-1">
              Domain: {proposalData?.domain || 'General'}
            </div>
          </Card>
        </div>
      </div>

      {/* 3. Save Proposal Action Panel */}
      <Card className="border-indigo-100 bg-white">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div>
            <h4 className="text-sm font-bold text-slate-900 flex items-center gap-2">
              <Database className="w-4 h-4 text-indigo-600" />
              Save Evaluated Proposal to Database
            </h4>
            <p className="text-xs text-slate-500 mt-0.5">
              Permanently store the candidate proposal and its explainable AI novelty report in local SQLite storage.
            </p>
          </div>

          <div className="flex items-center gap-3 w-full sm:w-auto">
            {saveSuccess ? (
              <div className="flex flex-wrap items-center gap-3">
                <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200">
                  <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                  Saved as {saveSuccess.project_id}
                </span>
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => onViewInHistory(saveSuccess.project_id)}
                >
                  <span>View in History</span>
                  <ArrowRight className="w-3.5 h-3.5 ml-1" />
                </Button>
              </div>
            ) : (
              <Button
                variant="primary"
                size="md"
                icon={Save}
                loading={saveLoading}
                disabled={isDirtyAfterScoring}
                onClick={onSave}
                className="w-full sm:w-auto"
              >
                {saveLoading ? 'Saving...' : 'Save Proposal'}
              </Button>
            )}
          </div>
        </div>

        {/* Warning if fields modified after scoring */}
        {isDirtyAfterScoring && (
          <div className="mt-4 p-3 rounded-lg bg-amber-50 text-amber-900 border border-amber-200 text-xs flex items-center justify-between gap-3">
            <span className="flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0" />
              The proposal was edited after evaluation. You must re-evaluate before saving.
            </span>
            <Button size="sm" variant="secondary" onClick={onReevaluate} className="shrink-0 text-xs">
              Re-evaluate Now
            </Button>
          </div>
        )}

        {/* Save error display */}
        {saveError && (
          <Alert variant="danger" title="Save Failed" className="mt-4">
            {saveError}
          </Alert>
        )}
      </Card>

      {/* 4. Visualizations: Section Cosine Similarity & Model Contributions */}
      {topMatch && <SectionCharts topMatch={topMatch} />}

      {/* 5. Top 5 Similar Historical Projects */}
      <div className="space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-200 pb-3">
          <div>
            <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
              <Layers className="w-4 h-4 text-indigo-600" />
              Top {topMatches.length} Closest Historical Projects
            </h3>
            <p className="text-xs text-slate-500 mt-0.5">
              Ranked by multi-section similarity learned from the reference corpus.
            </p>
          </div>
          <span className="text-xs text-slate-400 font-mono">
            Candidate: &ldquo;{report.candidate_proposal_title.slice(0, 30)}...&rdquo;
          </span>
        </div>

        <div className="space-y-3">
          {topMatches.map((match, idx) => (
            <SimilarProjectCard
              key={match.historical_project_id || idx}
              match={match}
              rank={idx + 1}
              isTopMatch={idx === 0}
            />
          ))}
        </div>
      </div>

      {/* 6. On-Demand Groq-Powered Explainable AI (XAI) Panel */}
      <AIExplanationPanel
        proposalData={proposalData}
        report={report}
        scoringMode={report.scoring_mode}
      />
    </div>
  );
}
