import React from 'react';
import { Eye, ArrowUpRight, Inbox } from 'lucide-react';
import { Card } from '../ui/Card';
import { Button } from '../ui/Button';
import { Badge } from '../ui/Badge';
import { formatPercent, formatScore, formatDate, getTierBadgeProps } from '../../utils/formatters';

export function RecentProposalsTable({
  proposals,
  loading,
  onViewDetails,
  onViewAllHistory,
  onEvaluateNew,
}) {
  return (
    <Card
      title="Recent Saved Submissions"
      subtitle="Evaluated proposals stored in the local SQLite database."
      action={
        proposals && proposals.length > 0 && (
          <Button variant="ghost" size="sm" onClick={onViewAllHistory}>
            <span>View All History</span>
            <ArrowUpRight className="w-3.5 h-3.5 ml-1" />
          </Button>
        )
      }
    >
      {loading ? (
        <div className="py-12 text-center text-sm text-slate-500 animate-pulse">
          Loading recent submissions...
        </div>
      ) : !proposals || proposals.length === 0 ? (
        <div className="py-12 text-center flex flex-col items-center justify-center">
          <div className="w-12 h-12 rounded-full bg-slate-100 flex items-center justify-center text-slate-400 mb-3">
            <Inbox className="w-6 h-6" />
          </div>
          <h4 className="text-sm font-semibold text-slate-800">No proposals saved yet</h4>
          <p className="text-xs text-slate-500 max-w-sm mt-1 mb-4">
            Evaluate an academic proposal to view novelty metrics, similarity breakdown, and save it to the records.
          </p>
          <Button size="sm" variant="primary" onClick={onEvaluateNew}>
            Evaluate First Proposal
          </Button>
        </div>
      ) : (
        <div className="overflow-x-auto -mx-6">
          <table className="w-full text-left text-sm text-slate-600">
            <thead className="bg-slate-50/80 text-[11px] uppercase tracking-wider font-semibold text-slate-500 border-y border-slate-100">
              <tr>
                <th className="py-3 px-6">ID & Title</th>
                <th className="py-3 px-4">Domain</th>
                <th className="py-3 px-4">Novelty %</th>
                <th className="py-3 px-4">Assessment Tier</th>
                <th className="py-3 px-4">Max Similarity</th>
                <th className="py-3 px-4">Submitted</th>
                <th className="py-3 px-6 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 font-normal">
              {proposals.map((item) => {
                const tierProps = getTierBadgeProps(item.novelty_assessment_tier);
                return (
                  <tr key={item.project_id || item.id} className="hover:bg-slate-50/80 transition-colors">
                    <td className="py-3.5 px-6">
                      <div className="font-semibold text-slate-900 line-clamp-1 max-w-xs">
                        {item.project_title}
                      </div>
                      <div className="text-[11px] font-mono text-slate-400">
                        {item.project_id}
                      </div>
                    </td>
                    <td className="py-3.5 px-4 text-xs">
                      <span className="text-slate-700">{item.domain || 'General'}</span>
                    </td>
                    <td className="py-3.5 px-4">
                      <span className="font-semibold text-slate-900">
                        {formatPercent(item.novelty_score_pct)}
                      </span>
                    </td>
                    <td className="py-3.5 px-4">
                      <span
                        className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium border ${tierProps.bgClass}`}
                      >
                        <span className={`w-1.5 h-1.5 rounded-full ${tierProps.dotClass}`} />
                        {item.novelty_assessment_tier}
                      </span>
                    </td>
                    <td className="py-3.5 px-4 text-xs font-mono text-slate-600">
                      {formatScore(item.max_similarity_score)}
                    </td>
                    <td className="py-3.5 px-4 text-xs text-slate-500 whitespace-nowrap">
                      {formatDate(item.created_at)}
                    </td>
                    <td className="py-3.5 px-6 text-right">
                      <Button
                        size="sm"
                        variant="secondary"
                        onClick={() => onViewDetails(item.project_id)}
                        className="text-xs"
                      >
                        <Eye className="w-3.5 h-3.5 mr-1" />
                        Details
                      </Button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}
