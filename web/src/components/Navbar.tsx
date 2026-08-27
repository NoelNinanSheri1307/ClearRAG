import React, { useState, useEffect } from 'react';
import { Calculator, Play } from 'lucide-react';
import { FormulasModal } from './FormulasModal';

interface NavbarProps {
  onNavigateToDemo?: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({ onNavigateToDemo }) => {
  const [scrolled, setScrolled] = useState(false);
  const [isFormulasOpen, setIsFormulasOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 40);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const navLinks = [
    { label: 'Problem', href: '#problem' },
    { label: 'Architecture', href: '#architecture' },
    { label: 'Benchmark', href: '#benchmark' },
    { label: 'Comparison', href: '#comparison' },
    { label: 'Tradeoff', href: '#tradeoff' },
    { label: 'Charts', href: '#pareto' },
    { label: 'Statistics', href: '#statistics' },
    { label: 'Dictionary', href: '#dictionary' },
  ];

  return (
    <>
      <header
        className={`fixed top-0 left-0 right-0 z-40 transition-all duration-300 ${
          scrolled
            ? 'bg-background/90 backdrop-blur-md border-b border-border/80 py-3.5 shadow-lg'
            : 'bg-transparent py-5'
        }`}
      >
        <div className="max-w-7xl mx-auto px-6 flex items-center justify-between">
          {/* Clean Minimalist Brand */}
          <a href="#" className="flex items-center group">
            <span className="text-base font-semibold tracking-wide text-foreground font-mono">
              ClearRAG
            </span>
          </a>

          {/* Desktop Navigation with Times New Roman Font for Links */}
          <nav className="hidden lg:flex items-center gap-6">
            {navLinks.map((link) => (
              <a
                key={link.label}
                href={link.href}
                style={{ fontFamily: "'Times New Roman', Times, serif" }}
                className="text-sm text-foreground-muted hover:text-foreground transition-colors font-normal tracking-wide"
              >
                {link.label}
              </a>
            ))}
          </nav>

          {/* Actions: Formulas + Demonstration */}
          <div className="flex items-center gap-2.5">
            <button
              onClick={() => setIsFormulasOpen(true)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-surface-100 border border-border hover:border-accent-teal/40 text-xs font-mono text-foreground-muted hover:text-accent-teal transition-colors"
              title="View all mathematical formulas"
            >
              <Calculator className="w-3.5 h-3.5" />
              <span>Formulas</span>
            </button>

            {onNavigateToDemo && (
              <button
                onClick={onNavigateToDemo}
                className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg bg-accent-teal text-background font-mono text-xs font-semibold hover:bg-accent-teal/90 transition-colors shadow-sm"
              >
                <Play className="w-3 h-3 fill-current" />
                <span>Interactive Demo</span>
              </button>
            )}
          </div>
        </div>
      </header>

      {/* Mathematical Formulas Dialog Modal */}
      <FormulasModal
        isOpen={isFormulasOpen}
        onClose={() => setIsFormulasOpen(false)}
      />
    </>
  );
};
