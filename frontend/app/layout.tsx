import type { Metadata } from "next";
import { Toaster } from "react-hot-toast";
import { Providers } from "@/src/components/Providers";
import { AppShell } from "@/src/components/AppShell";
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
          <AppShell>{children}</AppShell>
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
