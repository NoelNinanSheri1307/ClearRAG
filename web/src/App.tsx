import React, { useState, useEffect } from 'react';
import { LandingPage } from './features/landing/LandingPage';
import { DemoPage } from './features/demo/DemoPage';

export const App: React.FC = () => {
  const [view, setView] = useState<'research' | 'demo'>(() => {
    return window.location.hash === '#demo' ? 'demo' : 'research';
  });

  useEffect(() => {
    const handleHashChange = () => {
      if (window.location.hash === '#demo') {
        setView('demo');
      } else {
        setView('research');
      }
    };
    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  const navigateToDemo = () => {
    window.location.hash = '#demo';
    setView('demo');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const navigateToResearch = () => {
    window.location.hash = '';
    setView('research');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return view === 'demo' ? (
    <DemoPage onBackToResearch={navigateToResearch} />
  ) : (
    <LandingPage onNavigateToDemo={navigateToDemo} />
  );
};

export default App;
