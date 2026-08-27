import React from 'react';
import { ArrowUpRight, Search, FileText, Layers, CheckCircle2 } from 'lucide-react';

export const RetrievalSection: React.FC = () => {
  return (
    <section className="py-24 px-6 border-t border-border/60 bg-surface-300/40">
      <div className="max-w-5xl mx-auto">
        <div className="mb-14">
          <div className="text-xs font-mono uppercase tracking-widest text-accent-cyan mb-3">
            05 / Retrieval Performance
          </div>
          <h2 className="text-3xl sm:text-5xl font-serif text-foreground font-normal tracking-tight mb-4">
            If Evidence Is Missed, <br />
            <span className="italic text-foreground-muted">Verification Cannot Recover It.</span>
          </h2>
          <p className="max-w-2xl text-base text-foreground-muted font-sans font-light leading-relaxed">
            Before an AI can verify claims, it must first retrieve the right Wikipedia passages. Standard RAG relies only on semantic vector search, missing crucial exact names. ClearRAG combines semantic search with exact keyword search to boost evidence retrieval to <strong>87.84%</strong>.
          </p>
        </div>

        {/* 3 Comparison Cards: Standard RAG vs ClearRAG */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
          {/* Standard RAG Retrieval */}
          <div className="p-7 rounded-2xl bg-surface-100 border border-border">
            <span className="text-xs font-mono uppercase text-foreground-muted block mb-2">
              Standard RAG (Dense Search)
            </span>
            <div className="text-4xl sm:text-5xl font-mono font-normal text-foreground mb-2">
              69.12%
            </div>
            <p className="text-xs text-foreground-muted font-sans leading-relaxed">
              Missed required evidence on 386 out of 1,250 questions due to vocabulary mismatch on rare names and specific numbers.
            </p>
          </div>

          {/* ClearRAG Hybrid Retrieval */}
          <div className="p-7 rounded-2xl bg-surface-100 border border-accent-cyan/40 bg-accent-cyan/5">
            <span className="text-xs font-mono uppercase text-accent-cyan block mb-2">
              ClearRAG (Hybrid Search)
            </span>
            <div className="text-4xl sm:text-5xl font-mono font-normal text-accent-cyan mb-2">
              87.84%
            </div>
            <p className="text-xs text-foreground-muted font-sans leading-relaxed">
              Successfully retrieved the required evidence on 1,098 out of 1,250 questions by combining meaning search with exact keyword matching.
            </p>
          </div>

          {/* Advantage */}
          <div className="p-7 rounded-2xl bg-surface-100 border border-border flex flex-col justify-between">
            <div>
              <span className="text-xs font-mono uppercase text-foreground-muted block mb-2">
                Retrieval Advantage
              </span>
              <div className="text-3xl sm:text-4xl font-mono font-normal text-foreground mb-2 flex items-center gap-2">
                +18.72 pp
                <ArrowUpRight className="w-6 h-6 text-accent-cyan" />
              </div>
              <p className="text-xs text-foreground-muted font-sans leading-relaxed">
                Reduced missed-evidence failures by 60.6% (from 386 down to 152), providing a solid foundation for the verification layer.
              </p>
            </div>
            <div className="mt-4 pt-3 border-t border-border/60 text-[11px] font-mono text-accent-cyan">
              Evaluated across all 1,250 benchmark queries
            </div>
          </div>
        </div>

        {/* Technical Retrieval Stack in Plain English */}
        <div className="p-7 rounded-2xl bg-surface-100 border border-border">
          <div className="text-xs font-mono uppercase text-accent-cyan tracking-wider mb-4 font-semibold">
            How ClearRAG's Hybrid Retrieval Works
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs font-sans text-foreground-muted">
            <div className="p-4 rounded-xl bg-surface-200/70 border border-border/60">
              <strong className="text-foreground font-medium block mb-1">
                1. Semantic Meaning Search (BGE Dense Vectors)
              </strong>
              <span>
                Converts questions and articles into mathematical vectors to find passages that share the same overall <em>meaning</em> or topic, even if they use different wording.
              </span>
            </div>

            <div className="p-4 rounded-xl bg-surface-200/70 border border-border/70">
              <strong className="text-foreground font-medium block mb-1">
                2. Exact Keyword Search (BM25)
              </strong>
              <span>
                Scans for exact matches of rare proper nouns, specific dates, or entity names (e.g. "Walter Hill", "1980") that neural vectors often overlook.
              </span>
            </div>

            <div className="p-4 rounded-xl bg-surface-200/70 border border-border/60">
              <strong className="text-foreground font-medium block mb-1">
                3. Rank Fusion (RRF)
              </strong>
              <span>
                Merges the top results from both semantic search and keyword search, giving the highest priority to passages that scored well in both methods.
              </span>
            </div>

            <div className="p-4 rounded-xl bg-surface-200/70 border border-border/60">
              <strong className="text-foreground font-medium block mb-1">
                4. Diversity Reranking
              </strong>
              <span>
                Ensures the AI doesn't get flooded with multiple duplicate paragraphs from the same Wikipedia page, maximizing the variety of factual clues retrieved.
              </span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
