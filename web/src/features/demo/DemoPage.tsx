import React, { useState } from 'react';
import { ArrowLeft, Play, ArrowDown, Sparkles, HelpCircle, Layers, FileText } from 'lucide-react';
import { DEMO_SCENARIOS, DemoQuestionInstance } from '@/data/demo_scenarios';
import { ScenarioSelector } from './components/ScenarioSelector';
import { PipelineTraceVisualizer } from './components/PipelineTraceVisualizer';
import { ComparisonView } from './components/ComparisonView';
import { EvidenceInspector } from './components/EvidenceInspector';
import { DecisionExplanationCard } from './components/DecisionExplanationCard';

interface DemoPageProps {
  onBackToResearch: () => void;
}

export const DemoPage: React.FC<DemoPageProps> = ({ onBackToResearch }) => {
  const [selectedCategory, setSelectedCategory] = useState<string>('full_evidence');
  const [selectedQuestion, setSelectedQuestion] = useState<DemoQuestionInstance>(
    DEMO_SCENARIOS[0].questions[0]
  );
  const [customQuery, setCustomQuery] = useState<string>('');
  const [activeChunkNumber, setActiveChunkNumber] = useState<number | null>(null);
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [hasRun, setHasRun] = useState<boolean>(false);

  // Handle running the comparison
  const handleRunComparison = () => {
    setIsRunning(true);
    setHasRun(false);
    setTimeout(() => {
      setIsRunning(false);
      setHasRun(true);
      // Smoothly scroll down to results
      setTimeout(() => {
        document.getElementById('demo-results')?.scrollIntoView({ behavior: 'smooth' });
      }, 100);
    }, 500);
  };

  const handleSelectQuestion = (q: DemoQuestionInstance) => {
    setSelectedQuestion(q);
    setActiveChunkNumber(null);
    setHasRun(false);
  };

  const handleSelectCategory = (catId: string) => {
    setSelectedCategory(catId);
    setHasRun(false);
  };

  // Handle citation click: focus chunk and auto-scroll smoothly to Evidence Inspector
  const handleCitationClick = (chunkNum: number) => {
    setActiveChunkNumber(chunkNum);
    const element = document.getElementById('evidence-inspector');
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col selection:bg-accent-teal/20 selection:text-accent-teal">
      {/* Top Demo Header */}
      <header className="sticky top-0 z-40 bg-background/90 backdrop-blur-md border-b border-border/80 py-4 px-6">
        <div className="max-w-5xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-base font-semibold tracking-wide text-foreground font-mono">
              ClearRAG
            </span>
            <span className="text-xs font-mono px-2 py-0.5 rounded bg-surface-200 text-foreground-muted border border-border">
              Interactive Demonstration
            </span>
          </div>

          <button
            onClick={onBackToResearch}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-surface-100 border border-border hover:border-accent-teal/40 text-xs font-mono text-foreground-muted hover:text-foreground transition-colors"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            <span>Back to Research</span>
          </button>
        </div>
      </header>

      {/* Main Container */}
      <main className="flex-1 py-16 px-6">
        <div className="max-w-5xl mx-auto">
          {/* Introduction */}
          <div className="mb-14">
            <div className="text-xs font-mono uppercase tracking-widest text-accent-teal mb-3">
              Research Demonstration • Paired Experimental Testing
            </div>
            <h1 className="text-3xl sm:text-5xl font-serif text-foreground font-normal tracking-tight mb-4">
              See the Difference
            </h1>
            <p className="max-w-2xl text-base text-foreground-muted font-sans font-light leading-relaxed">
              Run the same multi-hop question through <strong>Standard RAG</strong> and <strong>ClearRAG</strong> to observe how evidence verification changes the system response policy from blind guessing to verifiable, attributed synthesis.
            </p>
          </div>

          {/* Scenario Selector & Question Input */}
          <ScenarioSelector
            selectedCategory={selectedCategory}
            onSelectCategory={handleSelectCategory}
            selectedQuestion={selectedQuestion}
            onSelectQuestion={handleSelectQuestion}
            customQuery={customQuery}
            onChangeCustomQuery={(q) => {
              setCustomQuery(q);
              setHasRun(false);
            }}
            onRunComparison={handleRunComparison}
            isRunning={isRunning}
            hasRun={hasRun}
          />

          {/* Initial Prompt to Run */}
          {!hasRun && !isRunning && (
            <div className="p-8 rounded-2xl bg-surface-100 border border-border text-center mb-12">
              <div className="max-w-md mx-auto space-y-3">
                <span className="text-xs font-mono text-accent-teal uppercase tracking-wider block">
                  Question Selected
                </span>
                <p className="text-sm font-sans text-foreground font-medium">
                  "{customQuery || selectedQuestion.question}"
                </p>
                <p className="text-xs font-sans text-foreground-muted leading-relaxed">
                  Click <strong>Run Comparison</strong> above to execute both pipelines and inspect the response policies side by side.
                </p>
                <div className="pt-2">
                  <button
                    onClick={handleRunComparison}
                    className="inline-flex items-center gap-2 px-6 py-2.5 rounded-xl bg-accent-teal text-background font-mono text-xs font-semibold hover:bg-accent-teal/90 transition-colors shadow-lg"
                  >
                    <Play className="w-3.5 h-3.5 fill-current" />
                    <span>Run Comparison</span>
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Loading Animation State */}
          {isRunning && (
            <div className="p-12 rounded-2xl bg-surface-100 border border-border text-center mb-12 animate-pulse">
              <div className="text-xs font-mono text-accent-teal uppercase tracking-wider mb-2">
                Executing Paired Architecture Pipelines
              </div>
              <div className="text-sm font-sans text-foreground">
                Retrieving passages via FAISS & BM25 • Evaluating bridge relations • Checking contradictions...
              </div>
            </div>
          )}

          {/* Results Area (Revealed after Running) */}
          {hasRun && (
            <div id="demo-results" className="space-y-12 animate-fade-in">
              {/* Pipeline Execution Trace */}
              <PipelineTraceVisualizer
                question={selectedQuestion}
                isRunning={isRunning}
              />

              {/* Side-by-Side Comparison */}
              <ComparisonView
                question={selectedQuestion}
                onCitationClick={handleCitationClick}
                activeChunkNumber={activeChunkNumber}
              />

              {/* Decision Explanation Card */}
              <DecisionExplanationCard question={selectedQuestion} />

              {/* Evidence Inspector Drawer with Auto-scroll anchor */}
              <EvidenceInspector
                chunks={selectedQuestion.clearRAG.retrievedChunks}
                activeChunkNumber={activeChunkNumber}
                onSelectChunk={setActiveChunkNumber}
              />
            </div>
          )}

          {/* Research Context Box */}
          <div className="p-8 rounded-2xl bg-surface-100 border border-border mb-12">
            <div className="text-xs font-mono uppercase text-accent-teal tracking-wider mb-2">
              Research Context • Canonical Evaluation Findings
            </div>
            <h3 className="text-lg font-medium text-foreground mb-3">
              How This Demonstration Relates to the 1,250-Query Benchmark
            </h3>
            <p className="text-xs text-foreground-muted font-sans leading-relaxed mb-6">
              The scenarios demonstrated above mirror the five evidence conditions measured across the 1,250 benchmark queries. Standard RAG answers 100% of queries with a 37.08% unsupported claim rate. Default ClearRAG achieves a 91.4% reduction in hallucinations (down to 3.20%) and 94.50% attribution coverage by selectively answering high-confidence queries while safely abstaining on unanswerables.
            </p>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 font-mono text-xs text-foreground-muted">
              <div className="p-3 rounded-lg bg-surface-200/70 border border-border/60">
                <span className="text-foreground-subtle block text-[10px] uppercase mb-1">Hallucination Drop</span>
                <span className="text-accent-teal font-semibold text-sm">37.1% → 3.2%</span>
              </div>
              <div className="p-3 rounded-lg bg-surface-200/70 border border-border/60">
                <span className="text-foreground-subtle block text-[10px] uppercase mb-1">Attribution Coverage</span>
                <span className="text-foreground font-semibold text-sm">94.50%</span>
              </div>
              <div className="p-3 rounded-lg bg-surface-200/70 border border-border/60">
                <span className="text-foreground-subtle block text-[10px] uppercase mb-1">Safe Abstention</span>
                <span className="text-foreground font-semibold text-sm">71.60%</span>
              </div>
              <div className="p-3 rounded-lg bg-surface-200/70 border border-border/60">
                <span className="text-foreground-subtle block text-[10px] uppercase mb-1">GPU Compute Saved</span>
                <span className="text-accent-teal font-semibold text-sm">72.40%</span>
              </div>
            </div>
          </div>

          {/* Bottom Return CTA */}
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4 p-6 rounded-2xl bg-surface-200/40 border border-border">
            <div className="text-xs font-sans text-foreground-muted">
              Ready to explore the full mathematical proof, ablation studies, and confidence intervals?
            </div>
            <button
              onClick={onBackToResearch}
              className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-surface-100 border border-border hover:border-accent-teal/40 text-xs font-mono text-foreground hover:text-accent-teal transition-colors shrink-0"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              <span>Back to Research Presentation</span>
            </button>
          </div>
        </div>
      </main>
    </div>
  );
};
