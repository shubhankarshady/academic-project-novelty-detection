import React, { useState } from 'react';
import { Sidebar } from './Sidebar';
import { Header } from './Header';

export function Layout({
  children,
  currentTab,
  setCurrentTab,
  title,
  subtitle,
  healthData,
  healthLoading,
  onRefreshHealth,
}) {
  const [isMobileOpen, setIsMobileOpen] = useState(false);

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col md:flex-row text-slate-900">
      <Sidebar
        currentTab={currentTab}
        setCurrentTab={setCurrentTab}
        healthData={healthData}
        isMobileOpen={isMobileOpen}
        setIsMobileOpen={setIsMobileOpen}
      />

      <div className="flex-1 flex flex-col min-w-0">
        <Header
          title={title}
          subtitle={subtitle}
          healthData={healthData}
          healthLoading={healthLoading}
          onRefreshHealth={onRefreshHealth}
          onToggleMobileMenu={() => setIsMobileOpen((prev) => !prev)}
        />

        <main className="flex-1 p-4 sm:p-8 max-w-7xl w-full mx-auto">{children}</main>
      </div>
    </div>
  );
}
