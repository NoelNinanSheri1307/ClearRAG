export interface EvidenceChunk {
  id: string;
  chunkNumber: number;
  title: string;
  sourceType: 'original' | 'synthetic_conflict' | 'distractor';
  score: number;
  text: string;
  isSupporting: boolean;
}

export interface DemoQuestionInstance {
  id: string;
  condition: 'full_evidence' | 'partial_evidence' | 'unsupported' | 'conflict' | 'distractor_heavy';
  question: string;
  goldAnswer: string;
  standardRAG: {
    answer: string;
    decision: 'ANSWER';
    latencyMs: number;
    llmCalled: boolean;
    evidenceSupport: 'SUPPORTED' | 'PARTIAL' | 'UNSUPPORTED' | 'CONFLICT';
    grounding: {
      supportedClaimRate: number;
      unsupportedClaimRate: number;
      faithfulnessScore: number;
      attributionCoverage: number;
    };
    retrievedChunks: EvidenceChunk[];
  };
  clearRAG: {
    answer: string;
    decision: 'ANSWER' | 'ANSWER_WITH_CAVEAT' | 'ABSTAIN' | 'CONFLICT_ABSTENTION';
    latencyMs: number;
    llmCalled: boolean;
    generationSkipped: boolean;
    confidenceScore: number;
    attributionCoverage: number;
    attributionPrecision: number;
    supportedClaimRate: number;
    unsupportedClaimRate: number;
    faithfulnessScore: number;
    citations: Array<{ marker: string; chunkNumber: number; claimText: string }>;
    verificationDetails: {
      status: string;
      reason: string;
      supportedClaimsCount: number;
      unsupportedClaimsCount: number;
      contradictionsFound: number;
    };
    retrievedChunks: EvidenceChunk[];
  };
  decisionExplanation: string;
}

export interface ScenarioCategory {
  id: 'full_evidence' | 'partial_evidence' | 'unsupported' | 'conflict' | 'distractor_heavy';
  label: string;
  badge: string;
  description: string;
  expectedBehavior: string;
  questions: DemoQuestionInstance[];
}

export const DEMO_SCENARIOS: ScenarioCategory[] = [
  {
    id: 'full_evidence',
    label: 'Full Evidence',
    badge: 'Supported Domain (250 Benchmark Queries)',
    description: 'Complete multi-hop factual bridge passages are present in the retrieved context.',
    expectedBehavior: 'Standard RAG answers blindly; ClearRAG verifies all claims and synthesizes an answer with sentence-level citations.',
    questions: [
      {
        id: 'hotpot_5ac2ed41554299218029db96_full_evidence',
        condition: 'full_evidence',
        question: 'Which genus has more species, Monstera or Cercis?',
        goldAnswer: 'Monstera',
        standardRAG: {
          answer: 'Monstera has more species than Cercis.',
          decision: 'ANSWER',
          latencyMs: 2490.0,
          llmCalled: true,
          evidenceSupport: 'SUPPORTED',
          grounding: {
            supportedClaimRate: 0.75,
            unsupportedClaimRate: 0.25,
            faithfulnessScore: 0.65,
            attributionCoverage: 0.0,
          },
          retrievedChunks: [
            {
              id: 'c1',
              chunkNumber: 1,
              title: 'Monstera (Wikipedia)',
              sourceType: 'original',
              score: 0.884,
              text: 'Monstera is a genus of about 50 species of flowering plants in the arum family, Araceae, native to tropical regions of the Americas.',
              isSupporting: true,
            },
            {
              id: 'c2',
              chunkNumber: 2,
              title: 'Cercis (Wikipedia)',
              sourceType: 'original',
              score: 0.842,
              text: 'Cercis is a genus of about 10 species in the subfamily Cercidoideae of the pea family Fabaceae, native to warm temperate regions.',
              isSupporting: true,
            },
            {
              id: 'c3',
              chunkNumber: 3,
              title: 'Rhaphidophora tetrasperma (Wikipedia)',
              sourceType: 'distractor',
              score: 0.612,
              text: 'Rhaphidophora tetrasperma (common name "Mini Monstera") is a species of plant in the family Araceae, genus Rhaphidophora.',
              isSupporting: false,
            },
          ],
        },
        clearRAG: {
          answer: 'Monstera has more species [1] than Cercis, which contains only about 10 species [2].',
          decision: 'ANSWER',
          latencyMs: 2412.3,
          llmCalled: true,
          generationSkipped: false,
          confidenceScore: 0.94,
          attributionCoverage: 1.0,
          attributionPrecision: 1.0,
          supportedClaimRate: 1.0,
          unsupportedClaimRate: 0.0,
          faithfulnessScore: 0.98,
          citations: [
            { marker: '[1]', chunkNumber: 1, claimText: 'Monstera has about 50 species' },
            { marker: '[2]', chunkNumber: 2, claimText: 'Cercis contains about 10 species' },
          ],
          verificationDetails: {
            status: 'FULLY_SUPPORTED',
            reason: 'All query claims verified against retrieved passages with cosine similarity θ=0.88 and 100% entity overlap.',
            supportedClaimsCount: 2,
            unsupportedClaimsCount: 0,
            contradictionsFound: 0,
          },
          retrievedChunks: [
            {
              id: 'c1',
              chunkNumber: 1,
              title: 'Monstera (Wikipedia)',
              sourceType: 'original',
              score: 0.912,
              text: 'Monstera is a genus of about 50 species of flowering plants in the arum family, Araceae, native to tropical regions of the Americas.',
              isSupporting: true,
            },
            {
              id: 'c2',
              chunkNumber: 2,
              title: 'Cercis (Wikipedia)',
              sourceType: 'original',
              score: 0.875,
              text: 'Cercis is a genus of about 10 species in the subfamily Cercidoideae of the pea family Fabaceae, native to warm temperate regions.',
              isSupporting: true,
            },
            {
              id: 'c3',
              chunkNumber: 3,
              title: 'Cercidiphyllum (Wikipedia)',
              sourceType: 'distractor',
              score: 0.584,
              text: 'Cercidiphyllum is a genus containing two species of plants, commonly called katsura.',
              isSupporting: false,
            },
          ],
        },
        decisionExplanation: 'Both systems answered because complete supporting evidence was retrieved. ClearRAG attached explicit sentence-level citations ([1], [2]) verifying that Monstera (50 species) exceeds Cercis (10 species).',
      },
      {
        id: 'hotpot_5a8d7e1055429941ae14dfd8_full_evidence',
        condition: 'full_evidence',
        question: 'Which is bigger, Griffon Bruxellois or English Mastiff?',
        goldAnswer: 'English Mastiff',
        standardRAG: {
          answer: 'The English Mastiff is much larger than the Griffon Bruxellois.',
          decision: 'ANSWER',
          latencyMs: 2490.0,
          llmCalled: true,
          evidenceSupport: 'SUPPORTED',
          grounding: {
            supportedClaimRate: 0.80,
            unsupportedClaimRate: 0.20,
            faithfulnessScore: 0.70,
            attributionCoverage: 0.0,
          },
          retrievedChunks: [
            {
              id: 'c1',
              chunkNumber: 1,
              title: 'English Mastiff (Wikipedia)',
              sourceType: 'original',
              score: 0.892,
              text: 'The English Mastiff is a breed of extremely large dog known for its massive size and heavy body, weighing up to 100 kg.',
              isSupporting: true,
            },
            {
              id: 'c2',
              chunkNumber: 2,
              title: 'Griffon Bruxellois (Wikipedia)',
              sourceType: 'original',
              score: 0.864,
              text: 'The Griffon Bruxellois is a toy breed of dog, typically weighing between 3.5 and 6 kg.',
              isSupporting: true,
            },
          ],
        },
        clearRAG: {
          answer: 'The English Mastiff is significantly larger [1] than the Griffon Bruxellois, which is a toy breed [2].',
          decision: 'ANSWER',
          latencyMs: 2380.1,
          llmCalled: true,
          generationSkipped: false,
          confidenceScore: 0.95,
          attributionCoverage: 1.0,
          attributionPrecision: 1.0,
          supportedClaimRate: 1.0,
          unsupportedClaimRate: 0.0,
          faithfulnessScore: 0.99,
          citations: [
            { marker: '[1]', chunkNumber: 1, claimText: 'English Mastiff is an extremely large dog up to 100 kg' },
            { marker: '[2]', chunkNumber: 2, claimText: 'Griffon Bruxellois is a toy breed weighing 3.5 to 6 kg' },
          ],
          verificationDetails: {
            status: 'FULLY_SUPPORTED',
            reason: 'Multi-hop entity comparison verified across both dog breed profiles.',
            supportedClaimsCount: 2,
            unsupportedClaimsCount: 0,
            contradictionsFound: 0,
          },
          retrievedChunks: [
            {
              id: 'c1',
              chunkNumber: 1,
              title: 'English Mastiff (Wikipedia)',
              sourceType: 'original',
              score: 0.931,
              text: 'The English Mastiff is a breed of extremely large dog known for its massive size and heavy body, weighing up to 100 kg.',
              isSupporting: true,
            },
            {
              id: 'c2',
              chunkNumber: 2,
              title: 'Griffon Bruxellois (Wikipedia)',
              sourceType: 'original',
              score: 0.902,
              text: 'The Griffon Bruxellois is a toy breed of dog, typically weighing between 3.5 and 6 kg.',
              isSupporting: true,
            },
          ],
        },
        decisionExplanation: 'Both systems answered accurately. ClearRAG confirmed the weight disparity through semantic verification and bounded the answer with traceable evidence links.',
      },
      {
        id: 'hotpot_5ab6644655429954757d327d_full_evidence',
        condition: 'full_evidence',
        question: 'What team competed in six competitions in their 116th season, and four in their 117th?',
        goldAnswer: 'Football Club Barcelona',
        standardRAG: {
          answer: 'Pune FC competed in six competitions in their 116th season.',
          decision: 'ANSWER',
          latencyMs: 2490.0,
          llmCalled: true,
          evidenceSupport: 'PARTIAL',
          grounding: {
            supportedClaimRate: 0.50,
            unsupportedClaimRate: 0.50,
            faithfulnessScore: 0.40,
            attributionCoverage: 0.0,
          },
          retrievedChunks: [
            {
              id: 'c1',
              chunkNumber: 1,
              title: 'Pune FC (Wikipedia)',
              sourceType: 'distractor',
              score: 0.742,
              text: 'Pune FC competed in domestic and regional football tournaments throughout several competitive campaigns.',
              isSupporting: false,
            },
          ],
        },
        clearRAG: {
          answer: 'FC Barcelona competed in six competitions during their 116th season [1] and four during their 117th season [2].',
          decision: 'ANSWER',
          latencyMs: 2440.0,
          llmCalled: true,
          generationSkipped: false,
          confidenceScore: 0.92,
          attributionCoverage: 1.0,
          attributionPrecision: 1.0,
          supportedClaimRate: 1.0,
          unsupportedClaimRate: 0.0,
          faithfulnessScore: 0.97,
          citations: [
            { marker: '[1]', chunkNumber: 1, claimText: '2015–16 FC Barcelona season was their 116th season, competing in 6 tournaments' },
            { marker: '[2]', chunkNumber: 2, claimText: '2016–17 FC Barcelona season was their 117th season, competing in 4 tournaments' },
          ],
          verificationDetails: {
            status: 'FULLY_SUPPORTED',
            reason: 'Hybrid BGE+BM25 retrieval brought both 116th and 117th season articles into context.',
            supportedClaimsCount: 2,
            unsupportedClaimsCount: 0,
            contradictionsFound: 0,
          },
          retrievedChunks: [
            {
              id: 'c1',
              chunkNumber: 1,
              title: '2015–16 FC Barcelona season (Wikipedia)',
              sourceType: 'original',
              score: 0.924,
              text: 'The 2015–16 FC Barcelona season was the club\'s 116th season in existence and the 85th consecutive season in the top flight of Spanish football, competing in six competitions.',
              isSupporting: true,
            },
            {
              id: 'c2',
              chunkNumber: 2,
              title: '2016–17 FC Barcelona season (Wikipedia)',
              sourceType: 'original',
              score: 0.895,
              text: 'The 2016–17 FC Barcelona season was the club\'s 117th season, during which they competed in four official competitions.',
              isSupporting: true,
            },
          ],
        },
        decisionExplanation: 'Standard RAG missed the bridge document due to single-stream vector search and hallucinated Pune FC. ClearRAG\'s hybrid BM25+dense retrieval found both season pages, allowing verified synthesis.',
      },
    ],
  },
  {
    id: 'partial_evidence',
    label: 'Partial Evidence',
    badge: 'Supported Domain (250 Benchmark Queries)',
    description: 'The retrieved context contains evidence for only part of the multi-hop query.',
    expectedBehavior: 'Standard RAG guesses the missing fact; ClearRAG synthesizes a qualified answer explicitly declaring what could not be verified.',
    questions: [
      {
        id: 'hotpot_partial_walter_hill_directors',
        condition: 'partial_evidence',
        question: 'Did the director of "The Warriors" also direct "Southern Comfort" and "Streets of Fire"?',
        goldAnswer: 'Yes, Walter Hill directed all three films.',
        standardRAG: {
          answer: 'Yes, Walter Hill directed The Warriors, Southern Comfort, and Streets of Fire in 1984.',
          decision: 'ANSWER',
          latencyMs: 2490.0,
          llmCalled: true,
          evidenceSupport: 'PARTIAL',
          grounding: {
            supportedClaimRate: 0.60,
            unsupportedClaimRate: 0.40,
            faithfulnessScore: 0.50,
            attributionCoverage: 0.0,
          },
          retrievedChunks: [
            {
              id: 'c1',
              chunkNumber: 1,
              title: 'The Warriors (film) (Wikipedia)',
              sourceType: 'original',
              score: 0.890,
              text: 'The Warriors is a 1979 American action thriller film directed by Walter Hill.',
              isSupporting: true,
            },
            {
              id: 'c2',
              chunkNumber: 2,
              title: 'Southern Comfort (1981 film) (Wikipedia)',
              sourceType: 'original',
              score: 0.855,
              text: 'Southern Comfort is a 1981 American action thriller film directed by Walter Hill.',
              isSupporting: true,
            },
          ],
        },
        clearRAG: {
          answer: 'Evidence confirms that Walter Hill directed "The Warriors" [1] and "Southern Comfort" [2]. However, the provided context does not contain sufficient evidence to verify whether he directed "Streets of Fire".',
          decision: 'ANSWER_WITH_CAVEAT',
          latencyMs: 2395.4,
          llmCalled: true,
          generationSkipped: false,
          confidenceScore: 0.68,
          attributionCoverage: 1.0,
          attributionPrecision: 1.0,
          supportedClaimRate: 1.0,
          unsupportedClaimRate: 0.0,
          faithfulnessScore: 0.95,
          citations: [
            { marker: '[1]', chunkNumber: 1, claimText: 'The Warriors was directed by Walter Hill' },
            { marker: '[2]', chunkNumber: 2, claimText: 'Southern Comfort was directed by Walter Hill' },
          ],
          verificationDetails: {
            status: 'PARTIALLY_SUPPORTED',
            reason: '2 of 3 multi-hop claims verified. "Streets of Fire" director passage was absent.',
            supportedClaimsCount: 2,
            unsupportedClaimsCount: 1,
            contradictionsFound: 0,
          },
          retrievedChunks: [
            {
              id: 'c1',
              chunkNumber: 1,
              title: 'The Warriors (film) (Wikipedia)',
              sourceType: 'original',
              score: 0.910,
              text: 'The Warriors is a 1979 American action thriller film directed by Walter Hill.',
              isSupporting: true,
            },
            {
              id: 'c2',
              chunkNumber: 2,
              title: 'Southern Comfort (1981 film) (Wikipedia)',
              sourceType: 'original',
              score: 0.875,
              text: 'Southern Comfort is a 1981 American action thriller film directed by Walter Hill.',
              isSupporting: true,
            },
          ],
        },
        decisionExplanation: 'Standard RAG guessed the unverified claim from parametric memory without hedging. ClearRAG recognized the missing third passage, triggering Caveat-Aware Generation.',
      },
      {
        id: 'hotpot_partial_swift_mother',
        condition: 'partial_evidence',
        question: 'Who is the mother of the American singer born in December 1989 who released the album "folklore"?',
        goldAnswer: 'Andrea Swift',
        standardRAG: {
          answer: 'Andrea Gardner Swift is the mother of Taylor Swift.',
          decision: 'ANSWER',
          latencyMs: 2490.0,
          llmCalled: true,
          evidenceSupport: 'PARTIAL',
          grounding: {
            supportedClaimRate: 0.50,
            unsupportedClaimRate: 0.50,
            faithfulnessScore: 0.50,
            attributionCoverage: 0.0,
          },
          retrievedChunks: [
            {
              id: 'c1',
              chunkNumber: 1,
              title: 'Taylor Swift (Wikipedia)',
              sourceType: 'original',
              score: 0.882,
              text: 'Taylor Alison Swift (born December 13, 1989) is an American singer-songwriter who released the 2020 album Folklore.',
              isSupporting: true,
            },
          ],
        },
        clearRAG: {
          answer: 'The retrieved context confirms Taylor Swift was born in December 1989 and released "folklore" [1]. However, the specific identity of her mother is not mentioned in the available documents.',
          decision: 'ANSWER_WITH_CAVEAT',
          latencyMs: 2310.0,
          llmCalled: true,
          generationSkipped: false,
          confidenceScore: 0.65,
          attributionCoverage: 1.0,
          attributionPrecision: 1.0,
          supportedClaimRate: 1.0,
          unsupportedClaimRate: 0.0,
          faithfulnessScore: 0.94,
          citations: [
            { marker: '[1]', chunkNumber: 1, claimText: 'Taylor Swift born Dec 1989 released Folklore' },
          ],
          verificationDetails: {
            status: 'PARTIALLY_SUPPORTED',
            reason: 'Singer identified from bridge evidence, but maternal relationship chunk is absent.',
            supportedClaimsCount: 1,
            unsupportedClaimsCount: 1,
            contradictionsFound: 0,
          },
          retrievedChunks: [
            {
              id: 'c1',
              chunkNumber: 1,
              title: 'Taylor Swift (Wikipedia)',
              sourceType: 'original',
              score: 0.920,
              text: 'Taylor Alison Swift (born December 13, 1989) is an American singer-songwriter who released the 2020 album Folklore.',
              isSupporting: true,
            },
          ],
        },
        decisionExplanation: 'Standard RAG pulled the mother\'s name from ungrounded parametric memory. ClearRAG bounded its generation strictly to the retrieved context, qualifying the response with an explicit evidence caveat.',
      },
    ],
  },
  {
    id: 'unsupported',
    label: 'Unsupported / No Evidence',
    badge: 'Unanswerable Domain (250 Benchmark Queries)',
    description: 'The target factual evidence is completely absent from the knowledge corpus.',
    expectedBehavior: 'Standard RAG fabricates a hallucinated answer; ClearRAG safely abstains and skips LLM generation entirely.',
    questions: [
      {
        id: 'hotpot_5a8888e35542997e5c09a5f6_unsupported',
        condition: 'unsupported',
        question: 'In what town is the university which the Bryant Bulldogs represent?',
        goldAnswer: 'Smithfield, Rhode Island',
        standardRAG: {
          answer: 'Smithfield, Rhode Island.',
          decision: 'ANSWER',
          latencyMs: 2490.0,
          llmCalled: true,
          evidenceSupport: 'UNSUPPORTED',
          grounding: {
            supportedClaimRate: 0.0,
            unsupportedClaimRate: 1.0,
            faithfulnessScore: 0.0,
            attributionCoverage: 0.0,
          },
          retrievedChunks: [
            {
              id: 'c1',
              chunkNumber: 1,
              title: 'Bulldog (Wikipedia)',
              sourceType: 'distractor',
              score: 0.520,
              text: 'The Bulldog is a medium-sized breed of dog commonly referred to as the English Bulldog or British Bulldog.',
              isSupporting: false,
            },
            {
              id: 'c2',
              chunkNumber: 2,
              title: 'Bryant Park (Wikipedia)',
              sourceType: 'distractor',
              score: 0.480,
              text: 'Bryant Park is a 9.6 acre privately managed public park located in the New York City borough of Manhattan.',
              isSupporting: false,
            },
          ],
        },
        clearRAG: {
          answer: 'I cannot provide a reliable answer to this question. The retrieved evidence does not contain sufficient information to support a factual response.',
          decision: 'ABSTAIN',
          latencyMs: 78.4,
          llmCalled: false,
          generationSkipped: true,
          confidenceScore: 0.12,
          attributionCoverage: 0.0,
          attributionPrecision: 1.0,
          supportedClaimRate: 1.0,
          unsupportedClaimRate: 0.0,
          faithfulnessScore: 1.0,
          citations: [],
          verificationDetails: {
            status: 'UNSUPPORTED',
            reason: 'Zero matching claims passed similarity threshold (max θ=0.21 < 0.65).',
            supportedClaimsCount: 0,
            unsupportedClaimsCount: 1,
            contradictionsFound: 0,
          },
          retrievedChunks: [
            {
              id: 'c1',
              chunkNumber: 1,
              title: 'Bulldog (Wikipedia)',
              sourceType: 'distractor',
              score: 0.520,
              text: 'The Bulldog is a medium-sized breed of dog commonly referred to as the English Bulldog or British Bulldog.',
              isSupporting: false,
            },
            {
              id: 'c2',
              chunkNumber: 2,
              title: 'Bryant Park (Wikipedia)',
              sourceType: 'distractor',
              score: 0.480,
              text: 'Bryant Park is a 9.6 acre privately managed public park located in the New York City borough of Manhattan.',
              isSupporting: false,
            },
          ],
        },
        decisionExplanation: 'No evidence existed in context. Standard RAG generated an answer from parametric memory without verification (100% unsupported). ClearRAG identified zero supported claims, safely abstained, and skipped LLM generation (saving 2.4s of GPU compute).',
      },
      {
        id: 'hotpot_5a832fdd55429954d2e2ec4f_unsupported',
        condition: 'unsupported',
        question: 'Who founded O Magazine that contained Elissa Schappell\'s second book of fiction?',
        goldAnswer: 'Oprah Winfrey and Hearst Communications',
        standardRAG: {
          answer: 'Newsweek and The Daily Beast founded the publication containing the excerpt.',
          decision: 'ANSWER',
          latencyMs: 2490.0,
          llmCalled: true,
          evidenceSupport: 'UNSUPPORTED',
          grounding: {
            supportedClaimRate: 0.0,
            unsupportedClaimRate: 1.0,
            faithfulnessScore: 0.0,
            attributionCoverage: 0.0,
          },
          retrievedChunks: [
            {
              id: 'c1',
              chunkNumber: 1,
              title: 'Elissa Schappell (Wikipedia)',
              sourceType: 'distractor',
              score: 0.610,
              text: 'Elissa Schappell is an American novelist, short story writer, and editor.',
              isSupporting: false,
            },
          ],
        },
        clearRAG: {
          answer: 'I cannot provide a reliable answer to this question. The retrieved evidence does not contain sufficient information to support a factual response.',
          decision: 'ABSTAIN',
          latencyMs: 82.1,
          llmCalled: false,
          generationSkipped: true,
          confidenceScore: 0.10,
          attributionCoverage: 0.0,
          attributionPrecision: 1.0,
          supportedClaimRate: 1.0,
          unsupportedClaimRate: 0.0,
          faithfulnessScore: 1.0,
          citations: [],
          verificationDetails: {
            status: 'UNSUPPORTED',
            reason: 'Bridge document regarding O Magazine founder was omitted from context.',
            supportedClaimsCount: 0,
            unsupportedClaimsCount: 1,
            contradictionsFound: 0,
          },
          retrievedChunks: [
            {
              id: 'c1',
              chunkNumber: 1,
              title: 'Elissa Schappell (Wikipedia)',
              sourceType: 'distractor',
              score: 0.610,
              text: 'Elissa Schappell is an American novelist, short story writer, and editor.',
              isSupporting: false,
            },
          ],
        },
        decisionExplanation: 'Standard RAG hallucinated "Newsweek". ClearRAG verified zero evidence support, bypassed the generator, and safely declared its inability to answer.',
      },
    ],
  },
  {
    id: 'conflict',
    label: 'Conflicting Evidence',
    badge: 'Unanswerable Domain (250 Benchmark Queries)',
    description: 'Retrieved sources contain contradictory dates, numbers, or mutually exclusive facts.',
    expectedBehavior: 'Standard RAG picks one contradictory source arbitrarily; ClearRAG flags the conflict and safely refuses to generate false consensus.',
    questions: [
      {
        id: 'hotpot_5a7df1095542990b8f503afe_conflict',
        condition: 'conflict',
        question: 'Who was born first out of Thomas Carr and Joyce Wieland?',
        goldAnswer: 'Thomas Carr',
        standardRAG: {
          answer: 'Thomas Carr was born first, as indicated by his birth date in 1904.',
          decision: 'ANSWER',
          latencyMs: 2490.0,
          llmCalled: true,
          evidenceSupport: 'CONFLICT',
          grounding: {
            supportedClaimRate: 0.50,
            unsupportedClaimRate: 0.50,
            faithfulnessScore: 0.35,
            attributionCoverage: 0.0,
          },
          retrievedChunks: [
            {
              id: 'c1',
              chunkNumber: 1,
              title: 'Thomas Carr (biography source A)',
              sourceType: 'original',
              score: 0.880,
              text: 'Thomas Carr was born on July 4, 1904, in Philadelphia, Pennsylvania.',
              isSupporting: true,
            },
            {
              id: 'c2',
              chunkNumber: 2,
              title: 'Thomas Carr (biography source B)',
              sourceType: 'synthetic_conflict',
              score: 0.875,
              text: 'Thomas Carr was born on July 4, 1937, in Philadelphia, Pennsylvania.',
              isSupporting: false,
            },
            {
              id: 'c3',
              chunkNumber: 3,
              title: 'Joyce Wieland (Wikipedia)',
              sourceType: 'original',
              score: 0.840,
              text: 'Joyce Wieland was born on June 30, 1930, in Toronto, Ontario.',
              isSupporting: true,
            },
          ],
        },
        clearRAG: {
          answer: 'I cannot provide a reliable answer to this question. The retrieved evidence contains conflicting information regarding the birth year of Thomas Carr (1904 vs 1937), making it impossible to determine a trustworthy response.',
          decision: 'CONFLICT_ABSTENTION',
          latencyMs: 94.2,
          llmCalled: false,
          generationSkipped: true,
          confidenceScore: 0.20,
          attributionCoverage: 0.0,
          attributionPrecision: 1.0,
          supportedClaimRate: 1.0,
          unsupportedClaimRate: 0.0,
          faithfulnessScore: 1.0,
          citations: [],
          verificationDetails: {
            status: 'CONFLICT_DETECTED',
            reason: 'Numeric temporal contradiction detected: Source A states 1904 while Source B states 1937 for Thomas Carr.',
            supportedClaimsCount: 1,
            unsupportedClaimsCount: 0,
            contradictionsFound: 1,
          },
          retrievedChunks: [
            {
              id: 'c1',
              chunkNumber: 1,
              title: 'Thomas Carr (Source A)',
              sourceType: 'original',
              score: 0.890,
              text: 'Thomas Carr was born on July 4, 1904, in Philadelphia, Pennsylvania.',
              isSupporting: true,
            },
            {
              id: 'c2',
              chunkNumber: 2,
              title: 'Thomas Carr (Source B)',
              sourceType: 'synthetic_conflict',
              score: 0.885,
              text: 'Thomas Carr was born on July 4, 1937, in Philadelphia, Pennsylvania.',
              isSupporting: false,
            },
            {
              id: 'c3',
              chunkNumber: 3,
              title: 'Joyce Wieland (Wikipedia)',
              sourceType: 'original',
              score: 0.840,
              text: 'Joyce Wieland was born on June 30, 1930, in Toronto, Ontario.',
              isSupporting: true,
            },
          ],
        },
        decisionExplanation: 'Retrieved sources directly contradicted each other on Thomas Carr\'s birth date (1904 vs 1937). Standard RAG arbitrarily chose 1904. ClearRAG detected the numeric divergence and safely abstained.',
      },
    ],
  },
  {
    id: 'distractor_heavy',
    label: 'Distractor-Heavy Context',
    badge: 'Distractor Domain (250 Benchmark Queries)',
    description: 'The correct bridge facts are buried inside 8+ noisy passages sharing high topical keywords.',
    expectedBehavior: 'Standard RAG gets misled by keyword distractors; ClearRAG\'s CrossScorer reranker isolates the genuine bridge facts.',
    questions: [
      {
        id: 'hotpot_5ae0e183554299422ee9953f_distractor',
        condition: 'distractor_heavy',
        question: 'Did Qionghai or Suining have a population of 658,798 in 2002?',
        goldAnswer: 'In 2002, Suining had a population of 658,798.',
        standardRAG: {
          answer: 'Qionghai had a population of 658,798 according to 2002 census reports.',
          decision: 'ANSWER',
          latencyMs: 2490.0,
          llmCalled: true,
          evidenceSupport: 'PARTIAL',
          grounding: {
            supportedClaimRate: 0.40,
            unsupportedClaimRate: 0.60,
            faithfulnessScore: 0.35,
            attributionCoverage: 0.0,
          },
          retrievedChunks: [
            {
              id: 'c1',
              chunkNumber: 1,
              title: 'Qionghai District (Wikipedia)',
              sourceType: 'distractor',
              score: 0.830,
              text: 'Qionghai is a county-level city in the east of Hainan province with expanding urban administration districts.',
              isSupporting: false,
            },
            {
              id: 'c2',
              chunkNumber: 2,
              title: 'Suining County, Jiangsu (Wikipedia)',
              sourceType: 'distractor',
              score: 0.810,
              text: 'Suining County is under the administration of Xuzhou, Jiangsu province.',
              isSupporting: false,
            },
          ],
        },
        clearRAG: {
          answer: 'Suining (Sichuan) had a population of 658,798 in 2002 [1], whereas Qionghai\'s population statistics differed [2].',
          decision: 'ANSWER',
          latencyMs: 2420.0,
          llmCalled: true,
          generationSkipped: false,
          confidenceScore: 0.89,
          attributionCoverage: 1.0,
          attributionPrecision: 1.0,
          supportedClaimRate: 1.0,
          unsupportedClaimRate: 0.0,
          faithfulnessScore: 0.96,
          citations: [
            { marker: '[1]', chunkNumber: 1, claimText: 'Suining population in 2002 was 658,798' },
            { marker: '[2]', chunkNumber: 2, claimText: 'Qionghai population records in 2002' },
          ],
          verificationDetails: {
            status: 'FULLY_SUPPORTED',
            reason: 'CrossScorer isolated the Sichuan Suining city statistics from Jiangsu county distractors.',
            supportedClaimsCount: 2,
            unsupportedClaimsCount: 0,
            contradictionsFound: 0,
          },
          retrievedChunks: [
            {
              id: 'c1',
              chunkNumber: 1,
              title: 'Suining (Wikipedia)',
              sourceType: 'original',
              score: 0.915,
              text: 'Suining is a prefecture-level city in the east of Sichuan province. In 2002, Suining had a registered urban population of 658,798.',
              isSupporting: true,
            },
            {
              id: 'c2',
              chunkNumber: 2,
              title: 'Qionghai (Wikipedia)',
              sourceType: 'distractor',
              score: 0.860,
              text: 'Qionghai had a total population of 483,217 recorded during provincial statistical surveys.',
              isSupporting: false,
            },
          ],
        },
        decisionExplanation: 'Standard RAG was deceived by overlapping municipal terms in distractor articles. ClearRAG\'s hybrid retrieval and entity reranker extracted the exact population match.',
      },
    ],
  },
];
