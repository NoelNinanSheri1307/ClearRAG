import React, { useState } from 'react';
import { X, Calculator, Search, Check, Copy } from 'lucide-react';

interface FormulaItem {
  id: string;
  name: string;
  category: 'Generation Quality' | 'Attribution & Grounding' | 'Decision & Safety' | 'Retrieval & Fusion' | 'Statistical Tests';
  formula: string;
  variables: string[];
  plainEnglish: string;
}

const FORMULAS: FormulaItem[] = [
  {
    id: 'em',
    name: 'Exact Match (EM)',
    category: 'Generation Quality',
    formula: 'EM = (Count of Normalised Answer == Normalised Reference) / Total Queries × 100%',
    variables: [
      'Normalisation: lowercased, punctuation stripped, articles (a, an, the) removed',
      'Answered-Instance EM: computed strictly over queries where the system generated an answer',
      'All-Instances EM: computed across all 1,250 queries (abstained queries count as 0.0)'
    ],
    plainEnglish: 'Measures the percentage of answers that match the gold-standard reference string character-for-character.'
  },
  {
    id: 'token_f1',
    name: 'Token F1 Score',
    category: 'Generation Quality',
    formula: 'Precision = |T_pred ∩ T_gold| / |T_pred|\nRecall = |T_pred ∩ T_gold| / |T_gold|\nF1 = 2 × (Precision × Recall) / (Precision + Recall)',
    variables: [
      'T_pred: Set of non-punctuation word tokens in generated response',
      'T_gold: Set of non-punctuation word tokens in reference answer'
    ],
    plainEnglish: 'The harmonic mean of precision (how many generated words were relevant) and recall (how many reference words were captured).'
  },
  {
    id: 'all_instances_f1',
    name: 'All-Instances vs Answered-Instance F1 Relation',
    category: 'Generation Quality',
    formula: 'F1_all_instances = F1_answered_instances × Answer_Coverage_Rate\nF1_all_instances = F1_answered_instances × (N_answered / N_total)',
    variables: [
      'N_answered: Number of queries where an answer was generated (345 for default ClearRAG)',
      'N_total: Total benchmark queries (1,250)'
    ],
    plainEnglish: 'Mathematical identity showing that abstaining scores 0.0 on all-instances, explaining the trade-off between strict safety and raw coverage.'
  },
  {
    id: 'unsupported_rate',
    name: 'Unsupported Claim Rate (Hallucinations)',
    category: 'Attribution & Grounding',
    formula: 'Unsupported Claim Rate = (Count of Sentences with No Supporting Context / Total Generated Sentences) × 100%',
    variables: [
      'Sentence is unsupported if no retrieved passage has semantic similarity θ ≥ 0.65 or entity overlap ≥ 0.35'
    ],
    plainEnglish: 'The direct measurement of factual hallucinations: the percentage of generated assertions lacking any factual evidence in retrieved context.'
  },
  {
    id: 'supported_rate',
    name: 'Supported Claim Rate',
    category: 'Attribution & Grounding',
    formula: 'Supported Claim Rate = 100% - Unsupported Claim Rate\nSupported Claim Rate = (Count of Factually Grounded Sentences / Total Generated Sentences) × 100%',
    variables: [
      'Measures overall factual grounding of the generated text'
    ],
    plainEnglish: 'The percentage of assertions in the generated response that are verifiable in the retrieved passages.'
  },
  {
    id: 'attribution_coverage',
    name: 'Attribution Coverage',
    category: 'Attribution & Grounding',
    formula: 'Attribution Coverage = (Sentences Containing Valid Citation Anchors [1], [2] / Total Sentences in Answer) × 100%',
    variables: [
      'Valid citation anchor: Regex pattern matching [\d+] pointing to an existing context passage index'
    ],
    plainEnglish: 'Measures how completely the AI includes explicit sentence-level citation references in its answers.'
  },
  {
    id: 'attribution_precision',
    name: 'Attribution Precision',
    category: 'Attribution & Grounding',
    formula: 'Attribution Precision = (Citations Accurately Grounding the Target Claim / Total Generated Citations) × 100%',
    variables: [
      'Tests whether chunk [i] cited by sentence s actually contains the facts asserted in sentence s'
    ],
    plainEnglish: 'The percentage of citation brackets that point to the correct, relevant passage rather than a random passage.'
  },
  {
    id: 'faithfulness_score',
    name: 'Faithfulness Score',
    category: 'Attribution & Grounding',
    formula: 'Faithfulness = (Claims Faithful to Context without Extraneous Speculation / Total Generated Claims) × 100%',
    variables: [
      'Evaluates consistency between generated text and source context'
    ],
    plainEnglish: 'The degree to which the model relies exclusively on provided context without inventing ungrounded details.'
  },
  {
    id: 'safe_abstention_rate',
    name: 'Safe Abstention Rate (on Unanswerables)',
    category: 'Decision & Safety',
    formula: 'Safe Abstention Rate = (Correct Abstentions on Unanswerable & Conflict Queries / Total Unanswerable Queries (500)) × 100%',
    variables: [
      'Unanswerable queries: 250 unsupported condition + 250 conflict condition = 500 queries'
    ],
    plainEnglish: 'The accuracy of the AI at recognizing when it does not have enough information and safely refusing to guess.'
  },
  {
    id: 'unsafe_answer_rate',
    name: 'Unsafe Answer Rate (on Unanswerables)',
    category: 'Decision & Safety',
    formula: 'Unsafe Answer Rate = (Erroneous Answers on Unanswerable Queries / Total Unanswerable Queries (500)) × 100%\nUnsafe Answer Rate = 100% - Safe Abstention Rate',
    variables: [
      'Directly measures dangerous overconfidence and failure to abstain'
    ],
    plainEnglish: 'How often the AI dangerously fabricated an answer for a query that had no supporting evidence.'
  },
  {
    id: 'answer_coverage_rate',
    name: 'Answer Coverage Rate',
    category: 'Decision & Safety',
    formula: 'Answer Coverage Rate = (Total Generated Answers / Total Benchmark Queries (1,250)) × 100%',
    variables: [
      'Standard RAG: 100% (1,250 / 1,250)',
      'ClearRAG Default: 27.60% (345 / 1,250)'
    ],
    plainEnglish: 'The proportion of incoming questions that the system attempts to answer rather than abstaining.'
  },
  {
    id: 'compute_saved',
    name: 'LLM Compute Saved',
    category: 'Decision & Safety',
    formula: 'Compute Saved = (Avoided LLM Generation Calls / Total Queries (1,250)) × 100%',
    variables: [
      'Avoided calls = Number of queries where verification layer triggered an early exit before calling the LLM'
    ],
    plainEnglish: 'The percentage of expensive GPU inference calls saved by stopping unanswerable queries before generation.'
  },
  {
    id: 'gold_retrieval_success',
    name: 'Gold Evidence Retrieval Success Rate',
    category: 'Retrieval & Fusion',
    formula: 'Gold Retrieval Success = (Queries with ALL Required Bridge Passages in Top-k / Total Queries (1,250)) × 100%',
    variables: [
      'Standard RAG: Top-5 Dense FAISS (69.12%)',
      'ClearRAG: Top-10 Hybrid RRF + Reranker (87.84%)'
    ],
    plainEnglish: 'How often the search stage successfully brought all required Wikipedia bridge facts into the context window.'
  },
  {
    id: 'rrf',
    name: 'Reciprocal Rank Fusion (RRF)',
    category: 'Retrieval & Fusion',
    formula: 'RRF_Score(document d) = ∑ (m ∈ {dense, bm25}) [ 1 / (k_rrf + rank_m(d)) ]',
    variables: [
      'k_rrf = 60 (standard smoothing constant)',
      'rank_dense(d): Position of document d in neural vector search (1-indexed)',
      'rank_bm25(d): Position of document d in keyword search (1-indexed)'
    ],
    plainEnglish: 'Combines the rankings of neural semantic search and keyword search into a single unified score prioritizing documents that rank high in both.'
  },
  {
    id: 'mcnemar_test',
    name: 'McNemar Paired Chi-Squared Statistic & 2x2 Contingency Matrix',
    category: 'Statistical Tests',
    formula: '2x2 Contingency Matrix (N = a + b + c + d = 1,250 queries):\n  * a = 432  (Both Standard RAG and ClearRAG made Safe decisions)\n  * b = 206  (Standard RAG Safe, ClearRAG Failed/Unsafe) [Discordant]\n  * c = 397  (ClearRAG Safe, Standard RAG Failed/Unsafe) [Discordant]\n  * d = 215  (Both Standard RAG and ClearRAG Failed/Unsafe)\n\nEdwards Continuity-Corrected Chi-Squared Formula:\nχ² = (|b - c| - 1)² / (b + c)\nχ² = (|206 - 397| - 1)² / (206 + 397) = (| -191 | - 1)² / 603 = 190² / 603 = 36,100 / 603 = 59.8673',
    variables: [
      'a = 432 (Concordant safe decisions for both systems)',
      'b = 206 (Discordant queries where Standard RAG was safe, ClearRAG failed)',
      'c = 397 (Discordant queries where ClearRAG was safe, Standard RAG failed/hallucinated)',
      'd = 215 (Concordant failure queries for both systems)',
      'Total queries: N = 432 + 206 + 397 + 215 = 1,250'
    ],
    plainEnglish: 'McNemar paired test evaluates discordant pairs (b vs c) on identical questions to determine if ClearRAG’s 397 safety wins over Standard RAG’s 206 wins is statistically significant.'
  },
  {
    id: 'p_value_calculation',
    name: 'Statistical p-Value Calculation',
    category: 'Statistical Tests',
    formula: 'p-value = P(X ≥ χ² | H₀) = 1 - F_χ²(χ² = 59.8673; df = 1)\np-value = ∫ [from 59.8673 to ∞] [ 1 / (√(2πx)) * e^(-x/2) ] dx\np-value = 1.0147 × 10⁻¹⁴  (≈ 0.0000000000000101)',
    variables: [
      'H₀ (Null Hypothesis): Both systems have identical safety performance (P(b) = P(c) = 0.5)',
      'df = 1 (1 Degree of Freedom for a 2x2 contingency table)',
      'F_χ²(x; df): Cumulative distribution function of the Chi-Squared distribution',
      'Significance threshold: α = 0.05 (and α = 0.01)',
      'Result: p = 1.01 × 10⁻¹⁴ << 0.001 (Null hypothesis decisively rejected)'
    ],
    plainEnglish: 'The p-value calculates the exact mathematical probability that ClearRAG achieved its safety advantage purely by random chance. A value of 1.01 × 10⁻¹⁴ means there is virtually zero chance this result was a coincidence.'
  },
  {
    id: 'odds_ratio',
    name: 'Statistical Odds Ratio',
    category: 'Statistical Tests',
    formula: 'Odds Ratio = c / b\nOdds Ratio = 397 / 206 = 1.9272 ≈ 1.93x',
    variables: [
      'c = 397 (ClearRAG safe discordant wins)',
      'b = 206 (Standard RAG safe discordant wins)',
      '95% Confidence Interval for Odds Ratio: [1.62, 2.30]'
    ],
    plainEnglish: 'Shows that on questions where the two systems disagree, ClearRAG is 1.93 times more likely to make the correct/safe decision than Standard RAG.'
  }
];

interface FormulasModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const FormulasModal: React.FC<FormulasModalProps> = ({ isOpen, onClose }) => {
  const [search, setSearch] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('All');
  const [copiedId, setCopiedId] = useState<string | null>(null);

  if (!isOpen) return null;

  const categories = ['All', 'Generation Quality', 'Attribution & Grounding', 'Decision & Safety', 'Retrieval & Fusion', 'Statistical Tests'];

  const filteredFormulas = FORMULAS.filter((f) => {
    const matchesSearch =
      f.name.toLowerCase().includes(search.toLowerCase()) ||
      f.formula.toLowerCase().includes(search.toLowerCase()) ||
      f.plainEnglish.toLowerCase().includes(search.toLowerCase());
    const matchesCat = selectedCategory === 'All' || f.category === selectedCategory;
    return matchesSearch && matchesCat;
  });

  const handleCopy = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 bg-black/80 backdrop-blur-md animate-fade-in"
      onClick={onClose}
    >
      <div
        className="relative max-w-4xl w-full max-h-[85vh] bg-surface-100 border border-border rounded-2xl shadow-2xl flex flex-col overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="p-6 border-b border-border/80 bg-surface-200/60 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-accent-teal/10 border border-accent-teal/30 flex items-center justify-center text-accent-teal">
              <Calculator className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-base sm:text-lg font-medium text-foreground">
                Mathematical Metric Formulations
              </h2>
              <p className="text-xs text-foreground-muted font-sans">
                Full mathematical formulas, variables, and plain-English definitions for all ClearRAG evaluation metrics.
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg text-foreground-muted hover:text-foreground hover:bg-surface-50 transition-colors"
            aria-label="Close"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Search & Categories */}
        <div className="p-4 sm:p-6 border-b border-border/60 bg-surface-100 flex flex-col sm:flex-row gap-3 items-stretch sm:items-center justify-between">
          <div className="flex flex-wrap gap-1.5">
            {categories.map((cat) => (
              <button
                key={cat}
                onClick={() => setSelectedCategory(cat)}
                className={`text-[11px] font-mono px-2.5 py-1 rounded-lg border transition-colors ${
                  selectedCategory === cat
                    ? 'bg-accent-teal/15 text-accent-teal border-accent-teal/30 font-medium'
                    : 'bg-surface-200/80 text-foreground-muted border-border hover:text-foreground'
                }`}
              >
                {cat}
              </button>
            ))}
          </div>

          <div className="relative shrink-0">
            <Search className="w-3.5 h-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-foreground-muted" />
            <input
              type="text"
              placeholder="Search formula..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-full sm:w-56 pl-8 pr-3 py-1.5 text-xs rounded-lg bg-surface-200/80 border border-border text-foreground placeholder:text-foreground-muted/60 focus:outline-none focus:border-accent-teal font-sans"
            />
          </div>
        </div>

        {/* Formula Cards List */}
        <div className="p-6 overflow-y-auto space-y-4 flex-1">
          {filteredFormulas.map((item) => (
            <div
              key={item.id}
              className="p-5 rounded-xl bg-surface-200/50 border border-border hover:border-accent-teal/30 transition-colors"
            >
              <div className="flex items-start justify-between gap-4 mb-2">
                <div className="flex items-center gap-2.5">
                  <h3 className="text-sm font-medium text-foreground">
                    {item.name}
                  </h3>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-surface-100 text-foreground-muted border border-border">
                    {item.category}
                  </span>
                </div>
                <button
                  onClick={() => handleCopy(item.formula, item.id)}
                  className="flex items-center gap-1 text-[11px] font-mono text-foreground-muted hover:text-accent-teal p-1 rounded hover:bg-surface-100 transition-colors"
                  title="Copy formula text"
                >
                  {copiedId === item.id ? (
                    <>
                      <Check className="w-3.5 h-3.5 text-accent-teal" />
                      <span className="text-accent-teal">Copied</span>
                    </>
                  ) : (
                    <>
                      <Copy className="w-3.5 h-3.5" />
                      <span>Copy</span>
                    </>
                  )}
                </button>
              </div>

              {/* Formula Block */}
              <div className="p-3.5 rounded-lg bg-surface-300 border border-border/80 text-accent-teal font-mono text-xs overflow-x-auto whitespace-pre-wrap mb-3">
                {item.formula}
              </div>

              {/* Plain English Meaning */}
              <p className="text-xs text-foreground font-sans leading-relaxed mb-2">
                <strong className="text-foreground-muted font-normal block text-[10px] font-mono uppercase tracking-wider mb-0.5">
                  Plain-English Interpretation
                </strong>
                {item.plainEnglish}
              </p>

              {/* Variables */}
              {item.variables.length > 0 && (
                <div className="pt-2 border-t border-border/40 space-y-1">
                  {item.variables.map((v, i) => (
                    <div key={i} className="text-[11px] text-foreground-muted font-sans flex items-start gap-1.5">
                      <span className="text-accent-teal font-mono">•</span>
                      <span>{v}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-border/80 bg-surface-200/40 flex items-center justify-between text-xs font-mono text-foreground-muted">
          <span>{filteredFormulas.length} Formulas Available</span>
          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg bg-surface-100 border border-border hover:bg-surface-50 text-foreground transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
