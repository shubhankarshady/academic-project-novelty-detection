import React from 'react';
import { Database, ShieldCheck, Cpu, ArrowRight } from 'lucide-react';
import { Card } from '../ui/Card';
import { Button } from '../ui/Button';

export function CorpusStatusCard({ healthData, onEvaluateClick }) {
  return (
    <Card className="bg-gradient-to-br from-indigo-900 to-slate-900 text-white border-indigo-950 shadow-md">
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
        <div className="space-y-3 max-w-2xl">
          <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-full text-xs font-semibold bg-indigo-500/20 text-indigo-200 border border-indigo-400/30">
            <ShieldCheck className="w-4 h-4 text-indigo-400" />
            Explainable AI (XAI) Model Active
          </div>

          <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-white">
            Academic Project Novelty Assessment Engine
          </h2>

          <p className="text-sm text-slate-300 leading-relaxed">
            Evaluates student and researcher project proposals across 8 structured text sections
            against a baseline reference corpus of{' '}
            <strong className="text-white font-semibold">
              {healthData?.reference_corpus_size ?? 600} historical projects
            </strong>
            . Provides transparent similarity attribution, section-level ranking, and verified
            keyword evidence.
          </p>

          <div className="flex flex-wrap items-center gap-4 pt-1 text-xs text-indigo-200">
            <span className="flex items-center gap-1.5">
              <Database className="w-4 h-4 text-indigo-400" />
              Reference Corpus: {healthData?.reference_corpus_size ?? 600} projects
            </span>
            <span className="flex items-center gap-1.5">
              <Cpu className="w-4 h-4 text-purple-400" />
              Additive Ridge Model & TF-IDF
            </span>
          </div>
        </div>

        <div className="shrink-0 w-full md:w-auto">
          <Button
            size="lg"
            variant="primary"
            onClick={onEvaluateClick}
            className="w-full md:w-auto bg-white text-indigo-900 hover:bg-slate-100 hover:text-indigo-950 font-semibold shadow-lg focus:ring-white"
          >
            <span>Evaluate New Proposal</span>
            <ArrowRight className="w-4 h-4 ml-1.5 text-indigo-700" />
          </Button>
        </div>
      </div>
    </Card>
  );
}
