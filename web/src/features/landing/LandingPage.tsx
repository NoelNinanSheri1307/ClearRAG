import React from 'react';
import { Navbar } from '@/components/Navbar';
import { Footer } from '@/components/Footer';
import { HeroSection } from './sections/HeroSection';
import { ProblemSection } from './sections/ProblemSection';
import { ResearchQuestionSection } from './sections/ResearchQuestionSection';
import { BenchmarkSection } from './sections/BenchmarkSection';
import { ArchitectureSection } from './sections/ArchitectureSection';
import { RetrievalSection } from './sections/RetrievalSection';
import { VerificationSection } from './sections/VerificationSection';
import { DecisionSection } from './sections/DecisionSection';
import { GenerationSection } from './sections/GenerationSection';
import { AttributionSection } from './sections/AttributionSection';
import { CaveatConflictSection } from './sections/CaveatConflictSection';
import { FinalComparisonSection } from './sections/FinalComparisonSection';
import { TradeoffSection } from './sections/TradeoffSection';
import { ParetoGallerySection } from './sections/ParetoGallerySection';
import { StatisticalValidationSection } from './sections/StatisticalValidationSection';
import { MetricDictionarySection } from './sections/MetricDictionarySection';
import { SynthesisSection } from './sections/SynthesisSection';

interface LandingPageProps {
  onNavigateToDemo?: () => void;
}

export const LandingPage: React.FC<LandingPageProps> = ({ onNavigateToDemo }) => {
  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col selection:bg-accent-teal/20 selection:text-accent-teal">
      <Navbar onNavigateToDemo={onNavigateToDemo} />
      <main className="flex-1">
        <HeroSection onNavigateToDemo={onNavigateToDemo} />
        <ProblemSection />
        <ResearchQuestionSection />
        <BenchmarkSection />
        <ArchitectureSection />
        <RetrievalSection />
        <VerificationSection />
        <DecisionSection />
        <GenerationSection />
        <AttributionSection />
        <CaveatConflictSection />
        <FinalComparisonSection />
        <TradeoffSection />
        <ParetoGallerySection />
        <StatisticalValidationSection />
        <MetricDictionarySection />
        <SynthesisSection />
      </main>
      <Footer />
    </div>
  );
};
