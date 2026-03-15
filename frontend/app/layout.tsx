import type { Metadata } from "next";
import { Toaster } from "react-hot-toast";
import { Providers } from "@/src/components/Providers";
import { Sidebar } from "@/src/components/layout/Sidebar";
import "./globals.css";

export const metadata: Metadata = {
  title: "Smart-Food Link",
  description: "AI-powered food waste reduction dashboard",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="light" suppressHydrationWarning>
      <body>
        <Providers>
          <div className="min-h-screen bg-background flex w-full overflow-hidden">
            <Sidebar />
            <main className="flex-1 w-full md:ml-64 pt-16 md:pt-0 min-h-screen">
              <div className="p-4 sm:p-6 lg:p-8 max-w-7xl mx-auto h-full">
                {children}
              </div>
            </main>
          </div>
          <Toaster
            position="top-right"
            toastOptions={{
              className: "border border-border/50 shadow-lg text-sm font-medium",
              style: {
                background: "hsl(var(--card))",
                color: "hsl(var(--foreground))",
                borderRadius: "12px",
              },
              success: {
                iconTheme: {
                  primary: "hsl(var(--success))",
                  secondary: "white",
                },
              },
              error: {
                iconTheme: {
                  primary: "hsl(var(--destructive))",
                  secondary: "white",
                },
              },
            }}
          />
        </Providers>
      </body>
    </html>
  );
}