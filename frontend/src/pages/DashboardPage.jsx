import React from 'react';
import { Database, FileText, CheckCircle2, Clock, Sparkles } from 'lucide-react';
import { StatCard } from '../components/dashboard/StatCard';
import { CorpusStatusCard } from '../components/dashboard/CorpusStatusCard';
import { RecentProposalsTable } from '../components/dashboard/RecentProposalsTable';

export function DashboardPage({
  healthData,
  healthLoading,
  recentProposals,
  proposalsLoading,
  onNavigateTab,
  onViewDetails,
}) {
  const refCorpusSize = healthData?.reference_corpus_size ?? 600;
  const submittedCount = healthData?.submitted_proposals_count ?? recentProposals?.length ?? 0;
  const isHealthy = healthData?.status === 'healthy';

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Top Welcome / Headline */}
      <div>
        <h2 className="text-2xl font-black text-slate-900 tracking-tight">
          System Overview & Metrics
        </h2>
        <p className="text-sm text-slate-500 mt-1">
          Monitor reference corpus integrity, model operational status, and recent proposal novelty scores.
        </p>
      </div>

      {/* KPI Stats Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        <StatCard
          title="Reference Corpus"
          value={refCorpusSize}
          subtitle="Baseline historical projects"
          icon={Database}
          color="indigo"
        />

        <StatCard
          title="Submitted Proposals"
          value={submittedCount}
          subtitle="User submissions in SQLite"
          icon={FileText}
          color="purple"
        />

        <StatCard
          title="Service Status"
          value={isHealthy ? 'Healthy' : 'Connecting'}
          subtitle={isHealthy ? 'Models loaded in memory' : 'Checking API status'}
          icon={CheckCircle2}
          color={isHealthy ? 'emerald' : 'amber'}
        />

        <StatCard
          title="Evaluation Pipeline"
          value="TF-IDF + Ridge"
          subtitle="8 sections & XAI evidence"
          icon={Sparkles}
          color="indigo"
        />
      </div>

      {/* Main Banner / Corpus Info */}
      <CorpusStatusCard
        healthData={healthData}
        onEvaluateClick={() => onNavigateTab('evaluate')}
      />

      {/* Recent Proposals Table */}
      <RecentProposalsTable
        proposals={recentProposals}
        loading={proposalsLoading}
        onViewDetails={onViewDetails}
        onViewAllHistory={() => onNavigateTab('history')}
        onEvaluateNew={() => onNavigateTab('evaluate')}
      />
    </div>
  );
}
