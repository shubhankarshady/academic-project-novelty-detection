import React, { useState } from 'react';
import {
  Search,
  Eye,
  ChevronLeft,
  ChevronRight,
  Database,
  Inbox,
  AlertCircle,
  RefreshCw,
} from 'lucide-react';
import { Card } from '../ui/Card';
import { Button } from '../ui/Button';
import { Badge } from '../ui/Badge';
import { Alert } from '../ui/Alert';
import { formatPercent, formatScore, formatDate, getTierBadgeProps } from '../../utils/formatters';

export function HistoryTable({
  proposals,
  total,
  limit,
  offset,
  loading,
  error,
  onPageChange,
  onRefresh,
  onViewDetails,
  onEvaluateNew,
}) {
  const [searchTerm, setSearchTerm] = useState('');

  // Local filter for search within fetched page
  const filteredProposals = (proposals || []).filter((item) => {
    if (!searchTerm.trim()) return true;
    const term = searchTerm.toLowerCase();
    return (
      (item.project_title && item.project_title.toLowerCase().includes(term)) ||
      (item.project_id && item.project_id.toLowerCase().includes(term)) ||
      (item.domain && item.domain.toLowerCase().includes(term))
    );
  });

  const currentPage = Math.floor(offset / limit) + 1;
  const totalPages = Math.max(1, Math.ceil(total / limit));

  return (
    <div className="space-y-6">
      {/* Corpus separation clarification notice */}
      <div className="bg-slate-100 border border-slate-200 rounded-xl p-4 text-xs text-slate-700 flex items-start gap-3">
        <Database className="w-4 h-4 text-indigo-600 shrink-0 mt-0.5" />
        <div>
          <strong className="text-slate-900">Database Integrity Note:</strong> Submitted proposals
          are stored in the user evaluations log (<code className="text-slate-800">proposals.db</code>) and{' '}
          <strong className="text-slate-900">do not</strong> enter or contaminate the 600 baseline
          historical comparison projects.
        </div>
      </div>

      <Card>
        {/* Table Search & Controls */}
        <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3 mb-4">
          <div className="relative flex-1 max-w-sm">
            <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Search by title, project ID, domain..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-9 pr-4 py-2 text-xs rounded-lg border border-slate-300 focus:outline-hidden focus:border-indigo-500 bg-slate-50/50 focus:bg-white transition-colors"
            />
          </div>

          <div className="flex items-center gap-2 self-end sm:self-auto">
            <span className="text-xs text-slate-500">
              Total records: <strong className="text-slate-800">{total}</strong>
            </span>
            <Button
              size="sm"
              variant="secondary"
              icon={RefreshCw}
              onClick={onRefresh}
              loading={loading}
              title="Refresh list"
            >
              Refresh
            </Button>
          </div>
        </div>

        {/* Error state */}
        {error && (
          <Alert variant="danger" title="Failed to Load History" className="mb-4">
            {error}
          </Alert>
        )}

        {/* Table Body */}
        {loading ? (
          <div className="py-16 text-center text-sm text-slate-500 animate-pulse">
            Loading proposal history from database...
          </div>
        ) : filteredProposals.length === 0 ? (
          <div className="py-16 text-center flex flex-col items-center justify-center">
            <div className="w-12 h-12 rounded-full bg-slate-100 flex items-center justify-center text-slate-400 mb-3">
              <Inbox className="w-6 h-6" />
            </div>
            <h4 className="text-sm font-semibold text-slate-800">
              {searchTerm ? 'No matching proposals found' : 'No submitted proposals yet'}
            </h4>
            <p className="text-xs text-slate-500 max-w-sm mt-1 mb-4">
              {searchTerm
                ? 'Try clearing the search query or adjusting your keywords.'
                : 'Proposals evaluated and explicitly saved will appear here with full history and XAI evidence.'}
            </p>
            {!searchTerm && (
              <Button size="sm" variant="primary" onClick={onEvaluateNew}>
                Evaluate New Proposal
              </Button>
            )}
          </div>
        ) : (
          <div className="overflow-x-auto -mx-6">
            <table className="w-full text-left text-sm text-slate-600">
              <thead className="bg-slate-50/80 text-[11px] uppercase tracking-wider font-semibold text-slate-500 border-y border-slate-100">
                <tr>
                  <th className="py-3 px-6">ID & Proposal Title</th>
                  <th className="py-3 px-4">Domain</th>
                  <th className="py-3 px-4">Novelty %</th>
                  <th className="py-3 px-4">Assessment Tier</th>
                  <th className="py-3 px-4">Max Similarity</th>
                  <th className="py-3 px-4">Submission Date</th>
                  <th className="py-3 px-6 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-normal">
                {filteredProposals.map((item) => {
                  const tierProps = getTierBadgeProps(item.novelty_assessment_tier);
                  return (
                    <tr
                      key={item.project_id || item.id}
                      className="hover:bg-slate-50/80 transition-colors"
                    >
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
                          View Details
                        </Button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        {/* Pagination footer */}
        {total > limit && (
          <div className="flex items-center justify-between border-t border-slate-100 pt-4 mt-2">
            <span className="text-xs text-slate-500">
              Showing page <strong className="text-slate-800">{currentPage}</strong> of{' '}
              <strong className="text-slate-800">{totalPages}</strong> ({total} total proposals)
            </span>

            <div className="flex items-center gap-1.5">
              <Button
                size="sm"
                variant="secondary"
                disabled={offset === 0 || loading}
                onClick={() => onPageChange(Math.max(0, offset - limit))}
              >
                <ChevronLeft className="w-4 h-4 mr-1" />
                Previous
              </Button>
              <Button
                size="sm"
                variant="secondary"
                disabled={offset + limit >= total || loading}
                onClick={() => onPageChange(offset + limit)}
              >
                Next
                <ChevronRight className="w-4 h-4 ml-1" />
              </Button>
            </div>
          </div>
        )}
      </Card>
    </div>
  );
}
