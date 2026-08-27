import React from 'react';
import { HelpCircle, Check, Search, ShieldCheck } from 'lucide-react';

export const ResearchQuestionSection: React.FC = () => {
  return (
    <section className="py-24 px-6 border-t border-border/60 bg-background relative">
      <div className="max-w-4xl mx-auto text-center">
        <div className="text-xs font-mono uppercase tracking-widest text-accent-teal mb-4">
          02 / The Research Question
        </div>

        {/* Narrative Callout */}
        <h2 className="text-3xl sm:text-5xl md:text-6xl font-serif text-foreground font-normal tracking-tight leading-tight mb-8">
          “Can a RAG system know <br className="hidden sm:block" />
          <span className="italic text-accent-teal">when it should answer?</span>”
        </h2>

        {/* Deep Scientific Formulation */}
        <div className="p-8 rounded-2xl bg-surface-100 border border-border text-left relative overflow-hidden">
          <div className="absolute top-0 right-0 w-32 h-32 bg-accent-teal/5 rounded-full blur-3xl pointer-events-none" />
          
          <h3 className="text-sm font-mono uppercase text-foreground-muted tracking-wider mb-4">
            Formal Hypothesis & Experimental Scope
          </h3>

          <p className="text-base sm:text-lg text-foreground font-sans font-light leading-relaxed mb-6">
            Can an explicit pipeline integrating <strong>semantic verification</strong> and an <strong>evidence-gated decision engine</strong> reduce unsupported hallucinated claims by refusing to answer unanswerable queries, while retaining verifiable attribution and useful answer quality when evidence is available?
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs font-sans text-foreground-muted border-t border-border/60 pt-6">
            <div className="flex items-start gap-2.5">
              <Check className="w-4 h-4 text-accent-teal shrink-0 mt-0.5" />
              <span><strong>Primary Target:</strong> Eliminate unsupported claims on missing and contradictory queries via deterministic abstention.</span>
            </div>
            <div className="flex items-start gap-2.5">
              <Check className="w-4 h-4 text-accent-teal shrink-0 mt-0.5" />
              <span><strong>Controlled Tradeoff:</strong> Quantify the exact cost in answer coverage and raw lexical generation scores (EM/F1).</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
