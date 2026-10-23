import React, { useState, useEffect } from 'react';
import { HistoryTable } from '../components/history/HistoryTable';
import { ProposalDetailModal } from '../components/history/ProposalDetailModal';
import { getProposals } from '../services/api';

export function HistoryPage({
  selectedProjectId,
  onClearSelectedProject,
  onNavigateTab,
}) {
  const [proposals, setProposals] = useState([]);
  const [total, setTotal] = useState(0);
  const [limit, setLimit] = useState(15);
  const [offset, setOffset] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Modal state
  const [activeModalId, setActiveModalId] = useState(selectedProjectId || null);

  useEffect(() => {
    if (selectedProjectId) {
      setActiveModalId(selectedProjectId);
    }
  }, [selectedProjectId]);

  useEffect(() => {
    loadProposals(offset);
  }, [offset, limit]);

  const loadProposals = async (currentOffset) => {
    setLoading(true);
    setError(null);
    try {
      const data = await getProposals(limit, currentOffset);
      setProposals(data.proposals || []);
      setTotal(data.total || 0);
    } catch (err) {
      setError(err.message || 'Failed to retrieve proposal history from database.');
    } finally {
      setLoading(false);
    }
  };

  const handlePageChange = (newOffset) => {
    setOffset(newOffset);
  };

  const handleViewDetails = (id) => {
    setActiveModalId(id);
  };

  const handleCloseModal = () => {
    setActiveModalId(null);
    if (onClearSelectedProject) {
      onClearSelectedProject();
    }
  };

  return (
    <div className="space-y-6 animate-fadeIn">
      <div>
        <h2 className="text-2xl font-black text-slate-900 tracking-tight">
          Proposal Evaluation History
        </h2>
        <p className="text-sm text-slate-500 mt-1">
          Review saved student and researcher proposals, historical novelty scores, and comprehensive explainability records.
        </p>
      </div>

      <HistoryTable
        proposals={proposals}
        total={total}
        limit={limit}
        offset={offset}
        loading={loading}
        error={error}
        onPageChange={handlePageChange}
        onRefresh={() => loadProposals(offset)}
        onViewDetails={handleViewDetails}
        onEvaluateNew={() => onNavigateTab('evaluate')}
      />

      <ProposalDetailModal
        projectId={activeModalId}
        isOpen={!!activeModalId}
        onClose={handleCloseModal}
      />
    </div>
  );
}
