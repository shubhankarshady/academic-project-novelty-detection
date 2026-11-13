import React, { useState } from 'react';
import { Sparkles, RotateCcw, FileText, AlertCircle, CheckCircle2, BookOpen, Cpu, SlidersHorizontal, Check } from 'lucide-react';
import { Card } from '../ui/Card';
import { Button } from '../ui/Button';
import { Alert } from '../ui/Alert';
import { Badge } from '../ui/Badge';
import { SECTION_CONFIG, SAMPLE_PROPOSALS } from '../../utils/formatters';

export function ProposalForm({
  formData,
  scoringMode = 'tfidf',
  onModeChange,
  onChange,
  onReset,
  onLoadSample,
  onSubmit,
  loading,
  error,
  isDirtyAfterScoring,
  hasScored,
}) {
  const [fieldErrors, setFieldErrors] = useState({});
  const [activeSampleIndex, setActiveSampleIndex] = useState(0);

  const validate = () => {
    const errors = {};
    for (const sec of SECTION_CONFIG) {
      if (sec.required) {
        const val = (formData[sec.key] || '').trim();
        if (!val) {
          errors[sec.key] = `${sec.label} is required.`;
        } else if (val.length < sec.minLength) {
          errors[sec.key] = `${sec.label} must be at least ${sec.minLength} characters (currently ${val.length}).`;
        }
      }
    }
    setFieldErrors(errors);
    return Object.keys(errors).length === 0;
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (validate()) {
      onSubmit();
    }
  };

  const handleFieldChange = (key, value) => {
    onChange(key, value);
    if (fieldErrors[key]) {
      setFieldErrors((prev) => {
        const copy = { ...prev };
        delete copy[key];
        return copy;
      });
    }
  };

  const handleSampleSelect = (e) => {
    const idx = parseInt(e.target.value, 10);
    setActiveSampleIndex(idx);
    onLoadSample(SAMPLE_PROPOSALS[idx].data);
    setFieldErrors({});
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {/* Top action / toolbar card */}
      <Card className="bg-gradient-to-r from-slate-900 via-indigo-950 to-slate-900 text-white border-slate-800">
        <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <Badge variant="indigo" size="sm" dot>
                8-Section Proposal Input
              </Badge>
              {formData._isSample && (
                <span className="px-2 py-0.5 rounded-full text-[11px] font-medium bg-amber-500/20 text-amber-300 border border-amber-400/30">
                  Demo Sample Preloaded
                </span>
              )}
            </div>
            <h3 className="text-base sm:text-lg font-bold text-white">
              Candidate Academic Proposal Details
            </h3>
            <p className="text-xs text-slate-300">
              Provide project details across all eight required academic sections to calculate novelty against the historical corpus.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2.5 w-full lg:w-auto">
            {/* Sample Selector */}
            <div className="flex items-center gap-2">
              <BookOpen className="w-4 h-4 text-indigo-400 hidden sm:block" />
              <select
                aria-label="Load demonstration sample proposal"
                value={activeSampleIndex}
                onChange={handleSampleSelect}
                className="bg-slate-800 text-slate-200 text-xs rounded-lg px-2.5 py-2 border border-slate-700 focus:outline-hidden focus:border-indigo-400 cursor-pointer"
              >
                {SAMPLE_PROPOSALS.map((s, idx) => (
                  <option key={s.name} value={idx}>
                    Preset: {s.name}
                  </option>
                ))}
              </select>
            </div>

            <Button
              type="button"
              variant="secondary"
              size="sm"
              icon={RotateCcw}
              onClick={onReset}
              className="bg-slate-800/80 text-slate-200 border-slate-700 hover:bg-slate-700 hover:text-white"
            >
              Reset
            </Button>

            <Button
              type="submit"
              variant="primary"
              size="md"
              icon={Sparkles}
              loading={loading}
              className="bg-indigo-500 hover:bg-indigo-600 text-white font-semibold shadow-md"
            >
              {loading ? 'Evaluating...' : 'Evaluate Novelty'}
            </Button>
          </div>
        </div>
      </Card>

      {/* Scoring Model Selector Card */}
      <div className="bg-white rounded-xl border border-slate-200/90 p-4 shadow-2xs">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-3">
          <div className="flex items-center gap-2">
            <SlidersHorizontal className="w-4 h-4 text-indigo-600" />
            <h4 className="text-sm font-bold text-slate-900">Scoring Model Architecture</h4>
            <span className="text-xs text-slate-500 hidden sm:inline">— Select evaluation engine</span>
          </div>
          <span className="text-[11px] text-slate-400 font-mono">
            Active: <strong className="text-indigo-600 font-bold">{scoringMode === 'tuned_hybrid' ? 'Tuned Hybrid (α=0.60)' : scoringMode === 'hybrid' ? 'Hybrid Baseline (α=0.50)' : 'TF-IDF Baseline'}</strong>
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {/* Mode 1: TF-IDF Baseline */}
          <button
            type="button"
            onClick={() => onModeChange && onModeChange('tfidf')}
            className={`p-3.5 rounded-lg border text-left transition-all cursor-pointer relative ${
              scoringMode === 'tfidf'
                ? 'border-indigo-600 bg-indigo-50/40 ring-1 ring-indigo-500'
                : 'border-slate-200 hover:border-slate-300 bg-white hover:bg-slate-50/50'
            }`}
          >
            <div className="flex items-start justify-between gap-2 mb-1">
              <span className="text-xs font-bold text-slate-900 flex items-center gap-1.5">
                TF-IDF Baseline
              </span>
              <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-slate-100 text-slate-600">
                Default
              </span>
            </div>
            <p className="text-[11px] text-slate-500 leading-snug">
              Production Ridge regression model using 8-section lexical TF-IDF n-grams.
            </p>
            <div className="mt-2 text-[10px] text-slate-400 font-mono">
              Weight: 100% Lexical
            </div>
          </button>

          {/* Mode 2: Hybrid Baseline (alpha 0.50) */}
          <button
            type="button"
            onClick={() => onModeChange && onModeChange('hybrid')}
            className={`p-3.5 rounded-lg border text-left transition-all cursor-pointer relative ${
              scoringMode === 'hybrid'
                ? 'border-purple-600 bg-purple-50/40 ring-1 ring-purple-500'
                : 'border-slate-200 hover:border-slate-300 bg-white hover:bg-slate-50/50'
            }`}
          >
            <div className="flex items-start justify-between gap-2 mb-1">
              <span className="text-xs font-bold text-slate-900 flex items-center gap-1.5">
                Hybrid Baseline (α=0.50)
              </span>
              <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-purple-100 text-purple-700">
                α=0.50
              </span>
            </div>
            <p className="text-[11px] text-slate-500 leading-snug">
              Lexical TF-IDF combined equally with all-MiniLM-L6-v2 dense embeddings.
            </p>
            <div className="mt-2 text-[10px] text-slate-400 font-mono">
              50% TF-IDF + 50% Semantic
            </div>
          </button>

          {/* Mode 3: Tuned Hybrid (alpha 0.60) */}
          <button
            type="button"
            onClick={() => onModeChange && onModeChange('tuned_hybrid')}
            className={`p-3.5 rounded-lg border text-left transition-all cursor-pointer relative ${
              scoringMode === 'tuned_hybrid'
                ? 'border-indigo-600 bg-indigo-50/50 ring-2 ring-indigo-500 shadow-2xs'
                : 'border-slate-200 hover:border-slate-300 bg-white hover:bg-slate-50/50'
            }`}
          >
            <div className="flex items-start justify-between gap-2 mb-1">
              <span className="text-xs font-bold text-slate-900 flex items-center gap-1.5">
                Tuned Hybrid (α=0.60)
              </span>
              <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-emerald-100 text-emerald-800">
                Calibrated
              </span>
            </div>
            <p className="text-[11px] text-slate-500 leading-snug">
              Phase 5 cross-validated hybrid model with candidate constrained weights.
            </p>
            <div className="mt-2 text-[10px] text-emerald-700 font-mono font-medium">
              60% TF-IDF + 40% Semantic
            </div>
          </button>
        </div>
      </div>

      {/* Dirty Warning if scored and edited */}
      {hasScored && isDirtyAfterScoring && (
        <Alert variant="warning" title="Proposal Modified Since Last Evaluation">
          You made changes to the proposal fields. The previous evaluation report is invalidated.
          Please click <strong>Evaluate Novelty</strong> again to refresh the analysis before saving.
        </Alert>
      )}

      {/* Global API error alert */}
      {error && (
        <Alert variant="danger" title="Evaluation Request Failed">
          {error}
        </Alert>
      )}

      {/* Input Sections */}
      <div className="space-y-5">
        {SECTION_CONFIG.map((sec, index) => {
          const val = formData[sec.key] || '';
          const hasError = !!fieldErrors[sec.key];
          const isShort = sec.rows === 1;

          return (
            <div
              key={sec.key}
              className={`bg-white rounded-xl border transition-all p-5 shadow-2xs ${
                hasError
                  ? 'border-rose-300 ring-1 ring-rose-300'
                  : 'border-slate-200/90 focus-within:border-indigo-500 focus-within:ring-1 focus-within:ring-indigo-500'
              }`}
            >
              <div className="flex flex-wrap items-center justify-between gap-2 mb-1.5">
                <label
                  htmlFor={sec.key}
                  className="text-sm font-semibold text-slate-900 flex items-center gap-1.5"
                >
                  <span className="w-5 h-5 rounded-md bg-slate-100 text-slate-600 text-xs flex items-center justify-center font-bold">
                    {index + 1}
                  </span>
                  {sec.label}
                  {sec.required && <span className="text-rose-500 text-sm">*</span>}
                </label>

                <div className="text-[11px] text-slate-400 font-mono">
                  {val.length} chars
                  {sec.minLength > 0 && ` (min ${sec.minLength})`}
                </div>
              </div>

              {sec.hint && (
                <p className="text-xs text-slate-500 mb-2.5">{sec.hint}</p>
              )}

              {isShort ? (
                <input
                  id={sec.key}
                  name={sec.key}
                  type="text"
                  value={val}
                  placeholder={sec.placeholder}
                  onChange={(e) => handleFieldChange(sec.key, e.target.value)}
                  className={`w-full px-3.5 py-2.5 rounded-lg text-sm text-slate-900 border transition-colors focus:outline-hidden ${
                    hasError
                      ? 'border-rose-300 bg-rose-50/20'
                      : 'border-slate-300 bg-slate-50/40 focus:bg-white focus:border-indigo-500'
                  }`}
                />
              ) : (
                <textarea
                  id={sec.key}
                  name={sec.key}
                  rows={sec.rows}
                  value={val}
                  placeholder={sec.placeholder}
                  onChange={(e) => handleFieldChange(sec.key, e.target.value)}
                  className={`w-full px-3.5 py-2.5 rounded-lg text-sm text-slate-900 border transition-colors focus:outline-hidden leading-relaxed ${
                    hasError
                      ? 'border-rose-300 bg-rose-50/20'
                      : 'border-slate-300 bg-slate-50/40 focus:bg-white focus:border-indigo-500'
                  }`}
                />
              )}

              {hasError && (
                <p className="mt-1.5 text-xs text-rose-600 flex items-center gap-1 font-medium">
                  <AlertCircle className="w-3.5 h-3.5" />
                  {fieldErrors[sec.key]}
                </p>
              )}
            </div>
          );
        })}
      </div>

      {/* Bottom evaluation button & notice */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-4 p-5 bg-white rounded-xl border border-slate-200">
        <div className="text-xs text-slate-500 leading-normal max-w-lg">
          <strong className="text-slate-700">Notice:</strong> Clicking Evaluate Novelty calculates
          similarity against the 600 reference projects and returns the XAI report. It{' '}
          <strong className="text-slate-700">does not save</strong> your proposal to the database
          until you review and explicitly click "Save Proposal".
        </div>

        <Button
          type="submit"
          variant="primary"
          size="lg"
          icon={Sparkles}
          loading={loading}
          className="w-full sm:w-auto"
        >
          {loading ? 'Analyzing Novelty...' : 'Evaluate Proposal'}
        </Button>
      </div>
    </form>
  );
}
