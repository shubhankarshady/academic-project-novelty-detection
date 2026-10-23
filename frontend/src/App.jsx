import React, { useState, useEffect } from 'react';
import { Layout } from './components/layout/Layout';
import { DashboardPage } from './pages/DashboardPage';
import { EvaluatePage } from './pages/EvaluatePage';
import { HistoryPage } from './pages/HistoryPage';
import { ProposalDetailModal } from './components/history/ProposalDetailModal';
import { getHealth, getProposals } from './services/api';

export function App() {
  const [currentTab, setCurrentTab] = useState('dashboard'); // 'dashboard' | 'evaluate' | 'history'
  const [healthData, setHealthData] = useState(null);
  const [healthLoading, setHealthLoading] = useState(true);

  // Recent proposals for dashboard preview
  const [recentProposals, setRecentProposals] = useState([]);
  const [proposalsLoading, setProposalsLoading] = useState(false);

  // Selected proposal for modal inspection
  const [selectedProjectId, setSelectedProjectId] = useState(null);

  // Load health & recent submissions on mount
  useEffect(() => {
    fetchHealthData();
    fetchRecentProposals();
  }, []);

  const fetchHealthData = async () => {
    setHealthLoading(true);
    try {
      const data = await getHealth();
      setHealthData(data);
    } catch {
      setHealthData({
        status: 'disconnected',
        reference_corpus_size: 600,
        submitted_proposals_count: 0,
        model_loaded: false,
        message: 'Could not connect to FastAPI backend.',
      });
    } finally {
      setHealthLoading(false);
    }
  };

  const fetchRecentProposals = async () => {
    setProposalsLoading(true);
    try {
      const data = await getProposals(5, 0);
      setRecentProposals(data.proposals || []);
    } catch {
      setRecentProposals([]);
    } finally {
      setProposalsLoading(false);
    }
  };

  const handleProposalSaved = () => {
    fetchHealthData();
    fetchRecentProposals();
  };

  const handleViewDetails = (projectId) => {
    setSelectedProjectId(projectId);
  };

  const getPageMeta = () => {
    switch (currentTab) {
      case 'evaluate':
        return {
          title: 'Evaluate Academic Proposal',
          subtitle: 'Assess proposal novelty and generate multi-section explainable AI report',
        };
      case 'history':
        return {
          title: 'Submitted Proposal History',
          subtitle: 'Past evaluated proposals stored in the local SQLite database',
        };
      default:
        return {
          title: 'Dashboard Overview',
          subtitle: 'Corpus status, baseline metrics, and recent proposal novelty scores',
        };
    }
  };

  const meta = getPageMeta();

  return (
    <Layout
      currentTab={currentTab}
      setCurrentTab={setCurrentTab}
      title={meta.title}
      subtitle={meta.subtitle}
      healthData={healthData}
      healthLoading={healthLoading}
      onRefreshHealth={fetchHealthData}
    >
      {currentTab === 'dashboard' && (
        <DashboardPage
          healthData={healthData}
          healthLoading={healthLoading}
          recentProposals={recentProposals}
          proposalsLoading={proposalsLoading}
          onNavigateTab={setCurrentTab}
          onViewDetails={handleViewDetails}
        />
      )}

      {currentTab === 'evaluate' && (
        <EvaluatePage
          onViewProposalDetails={(id) => {
            setSelectedProjectId(id);
            setCurrentTab('history');
          }}
          onProposalSaved={handleProposalSaved}
        />
      )}

      {currentTab === 'history' && (
        <HistoryPage
          selectedProjectId={selectedProjectId}
          onClearSelectedProject={() => setSelectedProjectId(null)}
          onNavigateTab={setCurrentTab}
        />
      )}

      {/* Global detail modal when selected from dashboard */}
      {currentTab === 'dashboard' && selectedProjectId && (
        <ProposalDetailModal
          projectId={selectedProjectId}
          isOpen={!!selectedProjectId}
          onClose={() => setSelectedProjectId(null)}
        />
      )}
    </Layout>
  );
}

export default App;
