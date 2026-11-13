import React, { useState } from 'react';
import {
  Sparkles,
  RotateCw,
  AlertTriangle,
  Lightbulb,
  Compass,
  CheckCircle2,
  BookOpen,
  Target,
  Cpu,
  Database,
  Layers,
  ListOrdered,
  ShieldCheck,
  ChevronRight,
  Info,
} from 'lucide-react';
import { Card } from '../ui/Card';
import { Button } from '../ui/Button';
import { Badge } from '../ui/Badge';
import { Alert } from '../ui/Alert';
import { generateAIExplanation } from '../../services/api';

export function AIExplanationPanel({
  proposalData,
  report,
  scoringMode,
}) {
  const [explanationData, setExplanationData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [regenWarning, setRegenWarning] = useState(null);

  if (!report || !proposalData) return null;

  const handleGenerate = async (forceRegenerate = false) => {
    setLoading(true);
    if (!forceRegenerate) {
      setError(null);
    }
    setRegenWarning(null);

    try {
      const res = await generateAIExplanation(proposalData, report, forceRegenerate);
      setExplanationData(res);
      setError(null);
      setRegenWarning(null);
    } catch (err) {
      const message = err.message || 'Failed to generate AI explanation.';
      if (explanationData && forceRegenerate) {
        // Preserve previous explanation and display alert
        setRegenWarning(`Regeneration failed: ${message}. Displaying previously generated analysis.`);
      } else {
        setError(message);
      }
    } finally {
      setLoading(false);
    }
  };

  const exp = explanationData?.explanation;

  return (
    <Card className="border-indigo-100 shadow-sm bg-white overflow-hidden mt-8 transition-all">
      {/* Header Banner */}
      <div className="bg-gradient-to-r from-indigo-900 via-indigo-800 to-slate-900 p-6 text-white">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="space-y-1">
            <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-indigo-500/30 text-indigo-200 border border-indigo-400/30">
              <Sparkles className="w-3.5 h-3.5 text-indigo-300" />
              <span>Groq LLaMA-3.3 Explainable AI (XAI)</span>
            </div>
            <h3 className="text-lg font-bold tracking-tight text-white flex items-center gap-2">
              Deep Semantic Proposal Interpretation & Research Guidance
            </h3>
            <p className="text-xs text-indigo-200/90 max-w-2xl">
              On-demand, multi-dimensional academic evaluation synthesizing research gaps,
              conceptual overlaps against reference initiatives, and a tailored roadmap.
            </p>
          </div>

          {/* Action Button: Generate or Regenerate */}
          <div className="shrink-0 flex items-center gap-3">
            {explanationData ? (
              <Button
                variant="secondary"
                size="md"
                icon={RotateCw}
                loading={loading}
                disabled={loading}
                onClick={() => handleGenerate(true)}
                className="bg-white/10 hover:bg-white/20 text-white border-white/20 hover:border-white/30 backdrop-blur-xs"
              >
                {loading ? 'Regenerating...' : 'Regenerate Explanation'}
              </Button>
            ) : (
              <button
                type="button"
                disabled={loading}
                onClick={() => handleGenerate(false)}
                className={`inline-flex items-center justify-center gap-2 px-5 py-2.5 rounded-xl font-bold text-sm shadow-md transition-all cursor-pointer ${
                  loading
                    ? 'bg-indigo-400 text-white cursor-not-allowed opacity-80'
                    : 'bg-gradient-to-r from-amber-400 to-amber-500 hover:from-amber-300 hover:to-amber-400 text-slate-950 hover:shadow-lg active:scale-98'
                }`}
              >
                <Sparkles className={`w-4 h-4 ${loading ? 'animate-spin' : 'text-amber-900'}`} />
                <span>{loading ? 'Analyzing your proposal with AI...' : '✨ Generate AI Explanation'}</span>
              </button>
            )}
          </div>
        </div>

        {/* Status badges when explanation loaded */}
        {explanationData && (
          <div className="mt-4 pt-3 border-t border-indigo-700/50 flex flex-wrap items-center gap-2 text-xs text-indigo-200">
            <span>Model: <strong className="text-white">{explanationData.model_used}</strong></span>
            <span>&bull;</span>
            <span>Mode: <strong className="text-white uppercase">{explanationData.scoring_mode}</strong> (α={explanationData.hybrid_alpha})</span>
            <span>&bull;</span>
            <span>Novelty: <strong className="text-white">{explanationData.overall_novelty_score_pct.toFixed(1)}%</strong> ({explanationData.novelty_assessment_tier})</span>
            {explanationData.cached && (
              <span className="ml-auto inline-flex items-center px-2 py-0.5 rounded text-[10px] font-semibold bg-indigo-500/30 text-indigo-200 border border-indigo-400/20">
                Cached Session
              </span>
            )}
          </div>
        )}
      </div>

      {/* Main Content Area */}
      <div className="p-6 space-y-6">
        {/* Warning if regeneration failed but previous explanation preserved */}
        {regenWarning && (
          <Alert variant="warning" title="Regeneration Notice">
            {regenWarning}
          </Alert>
        )}

        {/* Error state if generation failed with no prior data */}
        {error && !explanationData && (
          <div className="p-5 rounded-xl bg-rose-50 border border-rose-200 text-rose-900 space-y-3">
            <div className="flex items-start gap-3">
              <AlertTriangle className="w-5 h-5 text-rose-600 shrink-0 mt-0.5" />
              <div>
                <h4 className="text-sm font-bold text-rose-950">AI Explanation Service Unavailable</h4>
                <p className="text-xs text-rose-800 mt-1 leading-relaxed">{error}</p>
                <p className="text-xs text-rose-700 mt-2">
                  Tip: Ensure <code className="bg-rose-100 px-1.5 py-0.5 rounded font-mono text-rose-900">GROQ_API_KEY</code> is configured in your backend environment file or environment variables.
                </p>
              </div>
            </div>
            <div className="pt-2 flex justify-end">
              <Button size="sm" variant="secondary" onClick={() => handleGenerate(false)}>
                Try Again
              </Button>
            </div>
          </div>
        )}

        {/* Loading state indicator when request is running */}
        {loading && (
          <div className="py-12 flex flex-col items-center justify-center space-y-4 text-center">
            <div className="relative">
              <div className="w-14 h-14 rounded-full border-4 border-indigo-100 border-t-indigo-600 animate-spin" />
              <Sparkles className="w-6 h-6 text-indigo-600 absolute inset-0 m-auto animate-pulse" />
            </div>
            <div className="space-y-1">
              <h4 className="text-base font-bold text-slate-800">
                Analyzing your proposal with AI...
              </h4>
              <p className="text-xs text-slate-500 max-w-md">
                Interpreting multi-section similarity, contrasting against reference projects,
                detecting research gaps, and synthesizing a prioritized novelty action plan.
              </p>
            </div>
          </div>
        )}

        {/* Prompt to click button if not yet generated and not loading */}
        {!explanationData && !loading && !error && (
          <div className="py-10 text-center space-y-3 bg-slate-50/70 border border-dashed border-slate-200 rounded-xl">
            <div className="w-12 h-12 rounded-2xl bg-indigo-50 border border-indigo-100 flex items-center justify-center mx-auto text-indigo-600">
              <Sparkles className="w-6 h-6" />
            </div>
            <div className="space-y-1">
              <h4 className="text-sm font-bold text-slate-900">
                AI Explanation is Available On-Demand
              </h4>
              <p className="text-xs text-slate-500 max-w-md mx-auto">
                Click <strong>&ldquo;✨ Generate AI Explanation&rdquo;</strong> above to invoke Groq LLaMA-3.3.
                The AI will analyze the NLP findings, evaluate section overlaps, uncover research blind spots,
                and provide actionable recommendations tailored to your exact proposal.
              </p>
            </div>
          </div>
        )}

        {/* Structured 9-part explanation cards */}
        {exp && !loading && (
          <div className="space-y-6 animate-fadeIn">
            {/* 1. Overall Novelty Interpretation */}
            <div className="p-4 rounded-xl bg-indigo-50/80 border border-indigo-100 space-y-2">
              <div className="flex items-center gap-2 text-indigo-900 font-bold text-sm">
                <ShieldCheck className="w-4 h-4 text-indigo-600 shrink-0" />
                <span>1. Overall Novelty Interpretation</span>
              </div>
              <p className="text-xs sm:text-sm text-indigo-950 leading-relaxed font-medium">
                {exp.overall_novelty_interpretation}
              </p>
            </div>

            {/* 2 & 3: Missing Elements and Similarity Context */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
              {/* 2. What is Missing */}
              <div className="p-4 rounded-xl bg-amber-50/70 border border-amber-200/80 space-y-3">
                <div className="flex items-center gap-2 text-amber-900 font-bold text-sm">
                  <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0" />
                  <span>2. What Is Missing in the Proposal?</span>
                </div>
                <ul className="space-y-2 text-xs text-amber-950">
                  {exp.what_is_missing?.map((item, idx) => (
                    <li key={idx} className="flex items-start gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-amber-500 shrink-0 mt-1.5" />
                      <span>{item}</span>
                    </li>
                  ))}
                </ul>
              </div>

              {/* 3. Why Similar */}
              <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-2">
                <div className="flex items-center gap-2 text-slate-900 font-bold text-sm">
                  <Layers className="w-4 h-4 text-indigo-600 shrink-0" />
                  <span>3. Why Is It Similar to Existing Projects?</span>
                </div>
                <p className="text-xs text-slate-700 leading-relaxed">
                  {exp.why_similar}
                </p>
                {explanationData.top_match_id && (
                  <div className="text-[11px] text-slate-500 pt-1 font-mono">
                    Top Reference: <strong className="text-slate-800">{explanationData.top_match_id}</strong> &mdash; {explanationData.top_match_title}
                  </div>
                )}
              </div>
            </div>

            {/* 4. Section-Wise Overlap Explanation */}
            <div className="space-y-3">
              <h4 className="text-sm font-bold text-slate-900 flex items-center gap-2">
                <BookOpen className="w-4 h-4 text-indigo-600" />
                <span>4. Section-Wise Overlap Explanation</span>
              </h4>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                {exp.section_overlap_explanation?.map((item, idx) => {
                  const isHigh = item.overlap_level?.toLowerCase() === 'high';
                  const isLow = item.overlap_level?.toLowerCase() === 'low';
                  return (
                    <div
                      key={idx}
                      className="p-3.5 rounded-lg border bg-white shadow-2xs space-y-1.5 border-slate-200"
                    >
                      <div className="flex items-center justify-between gap-1">
                        <span className="text-xs font-bold text-slate-900">{item.section}</span>
                        <span
                          className={`text-[10px] font-bold px-2 py-0.5 rounded-full uppercase ${
                            isHigh
                              ? 'bg-rose-100 text-rose-800 border border-rose-200'
                              : isLow
                              ? 'bg-emerald-100 text-emerald-800 border border-emerald-200'
                              : 'bg-amber-100 text-amber-800 border border-amber-200'
                          }`}
                        >
                          {item.overlap_level}
                        </span>
                      </div>
                      <p className="text-xs text-slate-600 leading-relaxed">
                        {item.analysis}
                      </p>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* 5 & 6: Research Gaps and Specific Suggestions */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
              {/* 5. Potential Research Gaps */}
              <div className="p-4 rounded-xl bg-purple-50/70 border border-purple-200/80 space-y-3">
                <div className="flex items-center gap-2 text-purple-900 font-bold text-sm">
                  <Compass className="w-4 h-4 text-purple-600 shrink-0" />
                  <span>5. Potential Research Gaps in the Domain</span>
                </div>
                <ul className="space-y-2 text-xs text-purple-950">
                  {exp.potential_research_gaps?.map((gap, idx) => (
                    <li key={idx} className="flex items-start gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-purple-500 shrink-0 mt-1.5" />
                      <span>{gap}</span>
                    </li>
                  ))}
                </ul>
              </div>

              {/* 6. Specific Suggestions to Improve Novelty */}
              <div className="p-4 rounded-xl bg-emerald-50/70 border border-emerald-200/80 space-y-3">
                <div className="flex items-center gap-2 text-emerald-900 font-bold text-sm">
                  <Lightbulb className="w-4 h-4 text-emerald-600 shrink-0" />
                  <span>6. Specific Suggestions to Elevate Novelty</span>
                </div>
                <ul className="space-y-2 text-xs text-emerald-950">
                  {exp.specific_suggestions?.map((sug, idx) => (
                    <li key={idx} className="flex items-start gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 shrink-0 mt-1.5" />
                      <span>{sug}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>

            {/* 7. Potential Novel Research Directions */}
            <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-3">
              <div className="flex items-center gap-2 text-slate-900 font-bold text-sm">
                <Target className="w-4 h-4 text-indigo-600 shrink-0" />
                <span>7. Potential Novel Research Directions</span>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                {exp.potential_novel_directions?.map((dir, idx) => (
                  <div key={idx} className="p-2.5 rounded-lg bg-white border border-slate-200 text-xs text-slate-700 flex items-start gap-2">
                    <ChevronRight className="w-3.5 h-3.5 text-indigo-500 shrink-0 mt-0.5" />
                    <span>{dir}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* 8. Recommended Improvements Grid */}
            <div className="space-y-3">
              <h4 className="text-sm font-bold text-slate-900 flex items-center gap-2">
                <Cpu className="w-4 h-4 text-indigo-600" />
                <span>8. Recommended Improvements to Technical Sections</span>
              </h4>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {/* Objectives */}
                <div className="p-3.5 rounded-lg border border-slate-200 bg-white space-y-1">
                  <div className="text-xs font-bold uppercase tracking-wider text-indigo-600 flex items-center gap-1.5">
                    <Target className="w-3.5 h-3.5" />
                    <span>Objectives</span>
                  </div>
                  <p className="text-xs text-slate-700 leading-relaxed">
                    {exp.recommended_improvements?.objectives}
                  </p>
                </div>

                {/* Methodology */}
                <div className="p-3.5 rounded-lg border border-slate-200 bg-white space-y-1">
                  <div className="text-xs font-bold uppercase tracking-wider text-indigo-600 flex items-center gap-1.5">
                    <Layers className="w-3.5 h-3.5" />
                    <span>Proposed Methodology</span>
                  </div>
                  <p className="text-xs text-slate-700 leading-relaxed">
                    {exp.recommended_improvements?.proposed_methodology}
                  </p>
                </div>

                {/* Dataset */}
                <div className="p-3.5 rounded-lg border border-slate-200 bg-white space-y-1">
                  <div className="text-xs font-bold uppercase tracking-wider text-indigo-600 flex items-center gap-1.5">
                    <Database className="w-3.5 h-3.5" />
                    <span>Dataset & Benchmarks</span>
                  </div>
                  <p className="text-xs text-slate-700 leading-relaxed">
                    {exp.recommended_improvements?.dataset}
                  </p>
                </div>

                {/* Algorithms */}
                <div className="p-3.5 rounded-lg border border-slate-200 bg-white space-y-1">
                  <div className="text-xs font-bold uppercase tracking-wider text-indigo-600 flex items-center gap-1.5">
                    <Cpu className="w-3.5 h-3.5" />
                    <span>Algorithm & Architecture</span>
                  </div>
                  <p className="text-xs text-slate-700 leading-relaxed">
                    {exp.recommended_improvements?.algorithm_model}
                  </p>
                </div>
              </div>
            </div>

            {/* 9. Prioritized Action Plan */}
            <div className="p-4 rounded-xl bg-slate-900 text-white space-y-3">
              <div className="flex items-center gap-2 text-amber-300 font-bold text-sm">
                <ListOrdered className="w-4 h-4 text-amber-400 shrink-0" />
                <span>9. Prioritized Action Plan for Student</span>
              </div>
              <div className="space-y-2">
                {exp.prioritized_action_plan?.map((step, idx) => (
                  <div key={idx} className="flex items-start gap-3 text-xs sm:text-sm text-slate-200">
                    <span className="flex items-center justify-center w-5 h-5 rounded-full bg-indigo-500 text-white font-bold text-xs shrink-0 mt-0.5">
                      {idx + 1}
                    </span>
                    <span className="pt-0.5">{step}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </Card>
  );
}

