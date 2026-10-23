import React, { useState } from 'react';
import { Sparkles, FileText, BarChart3, AlertCircle } from 'lucide-react';
import { ProposalForm } from '../components/proposal/ProposalForm';
import { EvaluationReport } from '../components/report/EvaluationReport';
import { scoreProposal, saveProposal } from '../services/api';
import { SAMPLE_PROPOSALS } from '../utils/formatters';

const INITIAL_FORM = {
  project_title: '',
  domain: 'General',
  abstract: '',
  problem_statement: '',
  objectives: '',
  proposed_methodology: '',
  dataset: '',
  algorithm_model: '',
  technology_tools: '',
  _isSample: false,
};

export function EvaluatePage({
  onViewProposalDetails,
  onProposalSaved,
}) {
  const [activeSubTab, setActiveSubTab] = useState('input'); // 'input' | 'report'
  const [formData, setFormData] = useState(INITIAL_FORM);
  const [scoredFormData, setScoredFormData] = useState(null); // snapshot of form when scored
  const [evaluationReport, setEvaluationReport] = useState(null);
  const [isDirtyAfterScoring, setIsDirtyAfterScoring] = useState(false);

  // Loading & error states
  const [evalLoading, setEvalLoading] = useState(false);
  const [evalError, setEvalError] = useState(null);

  const [saveLoading, setSaveLoading] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(null);
  const [saveError, setSaveError] = useState(null);

  const handleFieldChange = (key, value) => {
    setFormData((prev) => ({
      ...prev,
      [key]: value,
      _isSample: false,
    }));

    if (evaluationReport) {
      setIsDirtyAfterScoring(true);
      setSaveSuccess(null);
    }
  };

  const handleReset = () => {
    if (window.confirm('Reset all proposal fields? Any unsaved evaluation results will be cleared.')) {
      setFormData(INITIAL_FORM);
      setScoredFormData(null);
      setEvaluationReport(null);
      setIsDirtyAfterScoring(false);
      setEvalError(null);
      setSaveSuccess(null);
      setSaveError(null);
      setActiveSubTab('input');
    }
  };

  const handleLoadSample = (sampleData) => {
    setFormData({
      ...sampleData,
      _isSample: true,
    });
    if (evaluationReport) {
      setIsDirtyAfterScoring(true);
      setSaveSuccess(null);
    }
  };

  // Evaluate candidate proposal via POST /api/score (does NOT save)
  const handleScoreProposal = async () => {
    setEvalLoading(true);
    setEvalError(null);
    setSaveError(null);
    setSaveSuccess(null);

    // Payload strips helper flags
    const payload = {
      project_title: formData.project_title.trim(),
      domain: (formData.domain || 'General').trim(),
      abstract: formData.abstract.trim(),
      problem_statement: formData.problem_statement.trim(),
      objectives: formData.objectives.trim(),
      proposed_methodology: formData.proposed_methodology.trim(),
      dataset: formData.dataset.trim(),
      algorithm_model: formData.algorithm_model.trim(),
      technology_tools: formData.technology_tools.trim(),
    };

    try {
      const report = await scoreProposal(payload);
      setEvaluationReport(report);
      setScoredFormData({ ...payload });
      setIsDirtyAfterScoring(false);
      setActiveSubTab('report');
    } catch (err) {
      setEvalError(err.message || 'Evaluation failed. Please review your inputs.');
    } finally {
      setEvalLoading(false);
    }
  };

  // Save proposal via POST /api/save
  const handleSaveProposal = async () => {
    if (!evaluationReport || !scoredFormData) {
      setSaveError('No valid evaluation report to save.');
      return;
    }

    if (isDirtyAfterScoring) {
      setSaveError('Proposal content was changed after scoring. Please re-evaluate before saving.');
      return;
    }

    setSaveLoading(true);
    setSaveError(null);

    try {
      const res = await saveProposal(scoredFormData, evaluationReport);
      setSaveSuccess(res);
      if (onProposalSaved) {
        onProposalSaved();
      }
    } catch (err) {
      setSaveError(err.message || 'Failed to save proposal to the database.');
    } finally {
      setSaveLoading(false);
    }
  };

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Top Navigation Subtabs */}
      <div className="flex border-b border-slate-200">
        <button
          type="button"
          onClick={() => setActiveSubTab('input')}
          className={`flex items-center gap-2 pb-3.5 px-1 sm:px-4 text-sm font-semibold cursor-pointer border-b-2 transition-all ${
            activeSubTab === 'input'
              ? 'border-indigo-600 text-indigo-600'
              : 'border-transparent text-slate-500 hover:text-slate-800'
          }`}
        >
          <FileText className="w-4 h-4" />
          <span>1. Proposal Input</span>
          {isDirtyAfterScoring && (
            <span className="w-2 h-2 rounded-full bg-amber-500" title="Modified since last score" />
          )}
        </button>

        <button
          type="button"
          disabled={!evaluationReport}
          onClick={() => setActiveSubTab('report')}
          className={`flex items-center gap-2 pb-3.5 px-1 sm:px-4 text-sm font-semibold transition-all ${
            !evaluationReport
              ? 'border-transparent text-slate-300 cursor-not-allowed'
              : activeSubTab === 'report'
              ? 'border-indigo-600 text-indigo-600 cursor-pointer'
              : 'border-transparent text-slate-500 hover:text-slate-800 cursor-pointer'
          }`}
        >
          <BarChart3 className="w-4 h-4" />
          <span>2. Novelty & Explainability Report</span>
          {evaluationReport && (
            <span className="px-1.5 py-0.2 rounded-full text-[10px] bg-indigo-100 text-indigo-700 font-bold">
              {evaluationReport.overall_novelty_score_pct.toFixed(0)}%
            </span>
          )}
        </button>
      </div>

      {/* Tab 1: Input Form */}
      {activeSubTab === 'input' && (
        <ProposalForm
          formData={formData}
          onChange={handleFieldChange}
          onReset={handleReset}
          onLoadSample={handleLoadSample}
          onSubmit={handleScoreProposal}
          loading={evalLoading}
          error={evalError}
          isDirtyAfterScoring={isDirtyAfterScoring}
          hasScored={!!evaluationReport}
        />
      )}

      {/* Tab 2: Evaluation Report */}
      {activeSubTab === 'report' && evaluationReport && (
        <EvaluationReport
          report={evaluationReport}
          proposalData={scoredFormData || formData}
          onSave={handleSaveProposal}
          saveLoading={saveLoading}
          saveSuccess={saveSuccess}
          saveError={saveError}
          isDirtyAfterScoring={isDirtyAfterScoring}
          onViewInHistory={onViewProposalDetails}
          onReevaluate={handleScoreProposal}
        />
      )}
    </div>
  );
}
