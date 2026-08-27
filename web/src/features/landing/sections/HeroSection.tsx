import React from 'react';
import { ArrowDown, Play } from 'lucide-react';

interface HeroSectionProps {
  onNavigateToDemo?: () => void;
}

export const HeroSection: React.FC<HeroSectionProps> = ({ onNavigateToDemo }) => {
  return (
    <section className="relative min-h-[88vh] flex flex-col justify-center items-center pt-32 pb-16 px-6 overflow-hidden bg-radial-gradient">
      {/* Subtle Grid Background */}
      <div className="absolute inset-0 bg-grid-pattern opacity-30 pointer-events-none" />

      {/* Narrative Container */}
      <div className="relative max-w-5xl mx-auto text-center z-10">
        {/* Category Lead */}
        <div className="text-xs font-mono uppercase tracking-widest text-foreground-subtle mb-6">
          Selective Retrieval-Augmented Generation
        </div>

        {/* Cinematic Headline */}
        <h1 className="text-4xl sm:text-6xl md:text-7xl font-serif font-normal text-foreground tracking-tight leading-[1.1] mb-6">
          When the evidence is not enough, <br className="hidden sm:block" />
          <span className="italic text-accent-teal">ClearRAG</span> can choose not to answer.
        </h1>

        {/* Precision Subtext */}
        <p className="max-w-2xl mx-auto text-base sm:text-lg text-foreground-muted leading-relaxed font-sans font-light mb-10">
          Conventional RAG systems retrieve documents and always generate an answer—even when the evidence is missing, partial, or contradictory. ClearRAG introduces an explicit verification and decision layer that detects unsupportable queries and safely abstains.
        </p>

        {/* Action Buttons: Demonstration + Methodology */}
        <div className="flex flex-wrap items-center justify-center gap-4 mb-14">
          {onNavigateToDemo && (
            <button
              onClick={onNavigateToDemo}
              className="flex items-center gap-2 px-6 py-3 rounded-xl bg-accent-teal text-background font-mono text-xs font-semibold hover:bg-accent-teal/90 transition-colors shadow-lg"
            >
              <Play className="w-3.5 h-3.5 fill-current" />
              <span>Explore Interactive Demonstration</span>
            </button>
          )}

          <a
            href="#problem"
            className="flex items-center gap-2 px-6 py-3 rounded-xl bg-surface-100 border border-border text-foreground font-mono text-xs font-medium hover:bg-surface-200 transition-colors"
          >
            <span>Read Research Report</span>
            <ArrowDown className="w-3.5 h-3.5 text-foreground-muted" />
          </a>
        </div>

        {/* Abstract Scientific Flow Visualization */}
        <div className="max-w-4xl mx-auto p-6 sm:p-8 rounded-xl bg-surface-200/50 border border-border/80 shadow-2xl backdrop-blur-sm mb-6">
          <div className="text-[11px] font-mono uppercase tracking-widest text-foreground-subtle mb-6 text-left flex items-center justify-between border-b border-border/60 pb-3">
            <span>Pipeline Architecture</span>
            <span className="text-foreground-muted font-mono">5-Stage Evaluation Flow</span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-5 gap-3 relative">
            {/* Step 1 */}
            <div className="p-4 rounded-lg bg-surface-100/70 border border-border flex flex-col items-center text-center">
              <span className="text-[10px] font-mono text-foreground-subtle mb-1">01</span>
              <span className="text-xs font-medium text-foreground mb-1">Question</span>
              <span className="text-[10px] text-foreground-muted">Natural Language</span>
            </div>

            {/* Step 2 */}
            <div className="p-4 rounded-lg bg-surface-100/70 border border-border flex flex-col items-center text-center">
              <span className="text-[10px] font-mono text-foreground-subtle mb-1">02</span>
              <span className="text-xs font-medium text-foreground mb-1">Hybrid Retrieval</span>
              <span className="text-[10px] text-foreground-muted font-mono">BGE + BM25 (k=10)</span>
            </div>

            {/* Step 3 */}
            <div className="p-4 rounded-lg bg-surface-100/70 border border-border flex flex-col items-center text-center">
              <span className="text-[10px] font-mono text-foreground-subtle mb-1">03</span>
              <span className="text-xs font-medium text-foreground mb-1">Verification</span>
              <span className="text-[10px] text-foreground-muted font-mono">Semantic & Conflict</span>
            </div>

            {/* Step 4 */}
            <div className="p-4 rounded-lg bg-surface-100/70 border border-border flex flex-col items-center text-center">
              <span className="text-[10px] font-mono text-foreground-subtle mb-1">04</span>
              <span className="text-xs font-medium text-foreground mb-1">Decision Layer</span>
              <span className="text-[10px] text-foreground-muted font-mono">Gate / Abstain</span>
            </div>

            {/* Step 5 */}
            <div className="p-4 rounded-lg bg-surface-100/70 border border-accent-teal/30 bg-accent-teal/5 flex flex-col items-center text-center">
              <span className="text-[10px] font-mono text-accent-teal mb-1">05</span>
              <span className="text-xs font-medium text-accent-teal mb-1">Grounded Output</span>
              <span className="text-[10px] text-accent-teal/80 font-mono">Attributed Answer</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
