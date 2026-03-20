import { useEffect } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import TopNav from "@/components/layout/TopNav";
import Toaster from "@/components/ui/Toaster";
import DashboardPage from "@/pages/DashboardPage";
import ProductionPlanningPage from "@/pages/ProductionPlanningPage";
import OperatorExecutionPage from "@/pages/OperatorExecutionPage";
import ReportsPage from "@/pages/ReportsPage";
import SettingsPage from "@/pages/SettingsPage";
import { useAppStore } from "@/store/appStore";

const queryClient = new QueryClient({
  defaultOptions: { queries: { staleTime: 30_000, retry: 1 } },
});

function ThemeProvider({ children }: { children: React.ReactNode }) {
  const theme = useAppStore((s) => s.theme);

  useEffect(() => {
    const html = document.documentElement;
    if (theme === "light") {
      html.classList.add("light");
    } else {
      html.classList.remove("light");
    }
  }, [theme]);

  return <>{children}</>;
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <ThemeProvider>
          <div className="min-h-screen bg-bg-base transition-colors duration-200">
            <TopNav />
            <main>
              <Routes>
                <Route path="/" element={<DashboardPage />} />
                <Route path="/planning" element={<ProductionPlanningPage />} />
                <Route path="/operator" element={<OperatorExecutionPage />} />
                <Route path="/reports" element={<ReportsPage />} />
                <Route path="/settings" element={<SettingsPage />} />
              </Routes>
            </main>
            <Toaster />
          </div>
        </ThemeProvider>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
