import React from 'react';
import { AlertCircle, Cpu, Database, Layers, ShieldAlert, Sparkles, HelpCircle, CheckCircle2 } from 'lucide-react';

export const ProblemSection: React.FC = () => {
  return (
    <section id="problem" className="py-24 px-6 border-t border-border/60 bg-surface-300/40">
      <div className="max-w-5xl mx-auto">
        {/* Section Header */}
        <div className="mb-14">
          <div className="text-xs font-mono uppercase tracking-widest text-accent-coral mb-3">
            01 / The Core Problem & Baseline Definition
          </div>
          <h2 className="text-3xl sm:text-5xl font-serif text-foreground font-normal tracking-tight mb-4">
            Conventional RAG Always Answers. <br />
            <span className="italic text-foreground-muted">Even When Evidence Does Not Exist.</span>
          </h2>
          <p className="max-w-2xl text-base text-foreground-muted font-sans font-light leading-relaxed">
            Standard Retrieval-Augmented Generation relies on a linear retrieve-then-generate paradigm. While effective when clear documents exist, it cannot determine whether what it retrieved is actually sufficient to support an answer.
          </p>
        </div>

        {/* Model Parity Highlight Box */}
        <div className="p-5 rounded-xl bg-accent-teal/5 border border-accent-teal/30 mb-8 flex items-start gap-3">
          <CheckCircle2 className="w-5 h-5 text-accent-teal shrink-0 mt-0.5" />
          <div className="text-xs font-sans text-foreground-muted leading-relaxed">
            <strong className="text-foreground block font-medium mb-0.5">Identical Model Stack & Controlled Experimental Fairness:</strong>
            Standard RAG and ClearRAG share the <strong>exact same underlying embedding model</strong> (BGE-small-en-v1.5) and the <strong>exact same generator LLM</strong> (Qwen 2.5 1.5B Instruct). The safety gains and 91.4% reduction in hallucinations are achieved purely through ClearRAG’s verification and gating architecture, with zero model-size advantage.
          </div>
        </div>

        {/* Standard RAG Architecture Card & Technical Stack */}
        <div className="p-8 rounded-2xl bg-surface-100 border border-border mb-12">
          <div className="flex items-center justify-between mb-6 pb-4 border-b border-border/60">
            <div>
              <span className="text-xs font-mono uppercase text-foreground-muted tracking-wider block">
                Experimental Control Baseline
              </span>
              <h3 className="text-lg font-medium text-foreground">
                Standard RAG Baseline Architecture
              </h3>
            </div>
            <span className="text-xs font-mono text-foreground-muted px-2.5 py-1 rounded bg-surface-200 border border-border">
              Retrieve → Generate
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            <div className="p-4 rounded-xl bg-surface-200/80 border border-border/60">
              <span className="text-[10px] font-mono uppercase text-foreground-subtle block mb-1">
                Dense Embedding Model
              </span>
              <strong className="text-xs font-mono text-foreground block mb-1">
                BAAI/bge-small-en-v1.5
              </strong>
              <p className="text-[11px] font-sans text-foreground-muted leading-relaxed">
                Converts questions and Wikipedia passages into 384-dimensional dense vectors representing semantic meaning.
              </p>
            </div>

            <div className="p-4 rounded-xl bg-surface-200/80 border border-border/60">
              <span className="text-[10px] font-mono uppercase text-foreground-subtle block mb-1">
                Vector Search Index (FAISS)
              </span>
              <strong className="text-xs font-mono text-foreground block mb-1">
                Top-5 Passages via FAISS
              </strong>
              <p className="text-[11px] font-sans text-foreground-muted leading-relaxed">
                <strong>FAISS</strong> (Facebook AI Similarity Search) calculates cosine similarity between the question vector and all passage vectors to instantly retrieve the 5 closest passages.
              </p>
            </div>

            <div className="p-4 rounded-xl bg-surface-200/80 border border-border/60">
              <span className="text-[10px] font-mono uppercase text-foreground-subtle block mb-1">
                Generator LLM & Decoding
              </span>
              <strong className="text-xs font-mono text-foreground block mb-1">
                Qwen 2.5 1.5B Instruct
              </strong>
              <p className="text-[11px] font-sans text-foreground-muted leading-relaxed">
                Greedy generation (temp=0.0) prompted to synthesize an answer directly from the 5 retrieved passages without verification.
              </p>
            </div>
          </div>

          <div className="p-4 rounded-xl bg-accent-coral/5 border border-accent-coral/30 flex items-start gap-3 text-xs font-sans text-foreground-muted">
            <AlertCircle className="w-4 h-4 text-accent-coral shrink-0 mt-0.5" />
            <span>
              <strong className="text-foreground">Always-Answer Policy:</strong> Standard RAG contains no verification step and no abstention mechanism. When context is missing, partial, or contradictory, the generator synthesizes plausible-sounding claims from parametric memory, producing a <strong>37.08% unsupported claim rate</strong>.
            </span>
          </div>
        </div>

        {/* The Failure Modes Matrix */}
        <div className="mb-6">
          <h3 className="text-sm font-mono uppercase tracking-wider text-accent-coral mb-4">
            The Four Evidence Failure Modes Studied
          </h3>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 font-sans text-xs">
            <div className="p-4 rounded-xl bg-surface-100 border border-border">
              <strong className="text-foreground block font-medium mb-1">1. Missing / Unsupported Evidence:</strong>
              <span className="text-foreground-muted">No relevant passages exist for the query. Standard RAG produces hallucinated facts.</span>
            </div>

            <div className="p-4 rounded-xl bg-surface-100 border border-border">
              <strong className="text-foreground block font-medium mb-1">2. Contradictory Evidence:</strong>
              <span className="text-foreground-muted">Retrieved sources disagree on facts (dates, names). Standard RAG picks one arbitrarily.</span>
            </div>

            <div className="p-4 rounded-xl bg-surface-100 border border-border">
              <strong className="text-foreground block font-medium mb-1">3. Partial Evidence:</strong>
              <span className="text-foreground-muted">Only partial evidence is found. Standard RAG guesses the missing facts without hedging.</span>
            </div>

            <div className="p-4 rounded-xl bg-surface-100 border border-border">
              <strong className="text-foreground block font-medium mb-1">4. Distractor-Heavy Context:</strong>
              <span className="text-foreground-muted">Topically similar but irrelevant passages deceive the LLM into answering off-target questions.</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
