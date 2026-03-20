"use client";

import { usePathname, useRouter } from "next/navigation";
import { useEffect } from "react";
import { Sidebar } from "./layout/Sidebar";
import { useAuth } from "./AuthContext";
import { Leaf } from "lucide-react";

const AUTH_ROUTES = ["/signin", "/signup"];

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const { isAuthenticated, isLoading } = useAuth();
  const isAuthPage = AUTH_ROUTES.includes(pathname);

  useEffect(() => {
    if (!isLoading && !isAuthenticated && !isAuthPage) {
      router.replace("/signin");
    }
    if (!isLoading && isAuthenticated && isAuthPage) {
      router.replace("/");
    }
  }, [isLoading, isAuthenticated, isAuthPage, router]);

  if (isLoading) return <LoadingScreen />;

  if (isAuthPage) {
    if (isAuthenticated) return null;
    return <>{children}</>;
  }

  if (!isAuthenticated) return null;

  return (
    <div className="min-h-screen bg-background flex w-full overflow-hidden">
      <Sidebar />
      <main className="flex-1 w-full md:ml-64 pt-16 md:pt-0 min-h-screen">
        <div className="p-4 sm:p-6 lg:p-8 max-w-7xl mx-auto h-full">
          {children}
        </div>
      </main>
    </div>
  );
}

function LoadingScreen() {
  return (
    <div className="min-h-screen bg-background flex items-center justify-center">
      <div className="text-center space-y-4">
        <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-primary to-emerald-500 flex items-center justify-center text-white shadow-lg mx-auto animate-pulse">
          <Leaf size={24} />
        </div>
        <p className="text-muted-foreground text-sm font-medium">Loading...</p>
      </div>
    </div>
  );
}
