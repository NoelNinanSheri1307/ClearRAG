import React from 'react';
import { GitBranch, ShieldCheck, CheckCircle2, XCircle, AlertOctagon, HelpCircle } from 'lucide-react';

export const DecisionSection: React.FC = () => {
  return (
    <section className="py-24 px-6 border-t border-border/60 bg-surface-300/40">
      <div className="max-w-5xl mx-auto">
        <div className="mb-14">
          <div className="text-xs font-mono uppercase tracking-widest text-accent-gold mb-3">
            07 / Decision & Abstention Layer
          </div>
          <h2 className="text-3xl sm:text-5xl font-serif text-foreground font-normal tracking-tight mb-4">
            Abstention Is Not an Error. <br />
            <span className="italic text-foreground-muted">It Is a Deliberate Safety Policy.</span>
          </h2>
          <p className="max-w-2xl text-base text-foreground-muted font-sans font-light leading-relaxed">
            When evidence is missing or contradictory, ClearRAG does not generate ungrounded answers. It executes a calibrated decision policy mapping verification status directly to system actions.
          </p>
        </div>

        {/* 4 Decision Branches Visual Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
          {/* Branch 1: Fully Supported */}
          <div className="p-6 rounded-2xl bg-surface-100 border border-accent-teal/40 bg-accent-teal/5 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-4">
                <span className="text-[11px] font-mono text-accent-teal uppercase">Status</span>
                <span className="text-xs font-mono font-medium text-foreground">100% Verified</span>
              </div>
              <div className="text-lg font-mono font-medium text-accent-teal mb-2">
                FULLY_SUPPORTED
              </div>
              <p className="text-xs text-foreground-muted font-sans leading-relaxed mb-4">
                All claims have strong evidence support in retrieved passages.
              </p>
            </div>
            <div className="pt-4 border-t border-border/60">
              <span className="text-[11px] font-mono text-foreground-muted block mb-1">Action</span>
              <span className="text-xs font-mono text-foreground font-medium px-2 py-1 rounded bg-surface-200 border border-border inline-block">
                ANSWER
              </span>
            </div>
          </div>

          {/* Branch 2: Partially Supported */}
          <div className="p-6 rounded-2xl bg-surface-100 border border-border flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-4">
                <span className="text-[11px] font-mono text-accent-cyan uppercase">Status</span>
                <span className="text-xs font-mono font-medium text-foreground">Partial Info</span>
              </div>
              <div className="text-lg font-mono font-medium text-foreground mb-2">
                PARTIALLY_SUPPORTED
              </div>
              <p className="text-xs text-foreground-muted font-sans leading-relaxed mb-4">
                Some facts are verified, but key bridge relations are missing.
              </p>
            </div>
            <div className="pt-4 border-t border-border/60">
              <span className="text-[11px] font-mono text-foreground-muted block mb-1">Action</span>
              <span className="text-xs font-mono text-foreground font-medium px-2 py-1 rounded bg-surface-200 border border-border inline-block">
                ANSWER_WITH_CAVEAT
              </span>
            </div>
          </div>

          {/* Branch 3: Unsupported */}
          <div className="p-6 rounded-2xl bg-surface-100 border border-border flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-4">
                <span className="text-[11px] font-mono text-accent-coral uppercase">Status</span>
                <span className="text-xs font-mono font-medium text-foreground">Zero Evidence</span>
              </div>
              <div className="text-lg font-mono font-medium text-foreground mb-2">
                UNSUPPORTED
              </div>
              <p className="text-xs text-foreground-muted font-sans leading-relaxed mb-4">
                No factual evidence found in context to support target claim.
              </p>
            </div>
            <div className="pt-4 border-t border-border/60">
              <span className="text-[11px] font-mono text-foreground-muted block mb-1">Action</span>
              <span className="text-xs font-mono text-accent-coral font-medium px-2 py-1 rounded bg-accent-coral/10 border border-accent-coral/20 inline-block">
                ABSTAIN
              </span>
            </div>
          </div>

          {/* Branch 4: Conflicting */}
          <div className="p-6 rounded-2xl bg-surface-100 border border-border flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-4">
                <span className="text-[11px] font-mono text-accent-amber uppercase">Status</span>
                <span className="text-xs font-mono font-medium text-foreground">Divergent Sources</span>
              </div>
              <div className="text-lg font-mono font-medium text-foreground mb-2">
                CONFLICTING
              </div>
              <p className="text-xs text-foreground-muted font-sans leading-relaxed mb-4">
                Retrieved chunks state contradictory attributes or numbers.
              </p>
            </div>
            <div className="pt-4 border-t border-border/60">
              <span className="text-[11px] font-mono text-foreground-muted block mb-1">Action</span>
              <span className="text-xs font-mono text-accent-amber font-medium px-2 py-1 rounded bg-accent-amber/10 border border-accent-amber/20 inline-block">
                CONFLICT_ABSTENTION
              </span>
            </div>
          </div>
        </div>

        {/* Tradeoff Explanation */}
        <div className="p-6 rounded-xl bg-surface-100 border border-border flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div className="text-xs font-sans text-foreground-muted">
            <strong className="text-foreground font-mono block text-[11px] uppercase mb-1">
              The Abstention Tradeoff
            </strong>
            <span>More strict abstention increases safety and cuts compute costs, but lowers nominal answer coverage.</span>
          </div>
          <span className="text-xs font-mono px-3 py-1.5 rounded-lg bg-surface-200 border border-border text-accent-teal shrink-0">
            71.60% Safe Abstention Achieved
          </span>
        </div>
      </div>
    </section>
  );
};
