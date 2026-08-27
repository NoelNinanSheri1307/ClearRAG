import React from 'react';
import { HelpCircle, Info, ArrowRight, ShieldCheck } from 'lucide-react';
import { DemoQuestionInstance } from '@/data/demo_scenarios';

interface DecisionExplanationCardProps {
  question: DemoQuestionInstance;
}

export const DecisionExplanationCard: React.FC<DecisionExplanationCardProps> = ({ question }) => {
  return (
    <div className="p-7 rounded-2xl bg-surface-100 border border-border mb-12">
      <div className="flex items-center gap-2 text-accent-teal mb-3">
        <Info className="w-4 h-4" />
        <span className="text-xs font-mono uppercase tracking-wider font-semibold">
          Scientific Decision Rationale • Why Are They Different?
        </span>
      </div>

      <p className="text-sm font-sans text-foreground leading-relaxed mb-4">
        {question.decisionExplanation}
      </p>

      <div className="p-4 rounded-xl bg-surface-200/60 border border-border/60 text-xs font-sans text-foreground-muted leading-relaxed">
        <strong className="text-foreground block font-medium mb-1">
          Research Takeaway:
        </strong>
        Standard RAG always assumes retrieved documents are sufficient to answer the prompt. ClearRAG explicitly separates <em>retrieval</em> from <em>decision gating</em>, verifying whether the retrieved bridge facts are valid and contradiction-free before invoking LLM synthesis.
      </div>
    </div>
  );
};
