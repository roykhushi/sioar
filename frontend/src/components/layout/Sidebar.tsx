"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import {
  LayoutDashboard, BrainCircuit, Building2,
  HeartHandshake, UserCircle, Menu, X, Leaf, LogOut
} from "lucide-react";
import { cn } from "@/lib/utils";
import { motion, AnimatePresence } from "framer-motion";
import { useAuth } from "@/src/components/AuthContext";
import axios from "axios";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8080";

const NAV_ITEMS = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard },
  { href: "/predict", label: "Risk Prediction", icon: BrainCircuit },
  { href: "/match", label: "Donation Match", icon: HeartHandshake },
  { href: "/ngos", label: "NGO Directory", icon: Building2 },
  { href: "/profile", label: "Profile", icon: UserCircle },
];

function ApiStatusIndicator() {
  const [isHealthy, setIsHealthy] = useState<boolean | null>(null);

  useEffect(() => {
    const check = () => {
      axios.get(`${API_BASE_URL}/`)
        .then(() => setIsHealthy(true))
        .catch(() => setIsHealthy(false));
    };
    check();
    const interval = setInterval(check, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-secondary/50 border border-border/50">
      <div className="relative flex h-2.5 w-2.5">
        {isHealthy && (
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-success opacity-40"></span>
        )}
        <span className={cn(
          "relative inline-flex rounded-full h-2.5 w-2.5",
          isHealthy === null ? "bg-muted-foreground" :
          isHealthy ? "bg-success" : "bg-destructive"
        )}></span>
      </div>
      <span className="text-xs font-medium text-muted-foreground">
        {isHealthy === null ? "Checking..." : isHealthy ? "API Connected" : "API Offline"}
      </span>
    </div>
  );
}

export function Sidebar() {
  const pathname = usePathname();
  const router = useRouter();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const { user, logout } = useAuth();

  useEffect(() => {
    setMobileMenuOpen(false);
  }, [pathname]);

  const handleLogout = () => {
    logout();
    router.push("/signin");
  };

  return (
    <>
      {/* Desktop Sidebar */}
      <aside className="hidden md:flex flex-col w-64 bg-card border-r border-border/60 fixed inset-y-0 z-20">
        <div className="h-20 flex items-center px-6 border-b border-border/40">
          <Link href="/" className="flex items-center gap-2 group cursor-pointer">
            <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-primary to-emerald-500 flex items-center justify-center text-white shadow-md shadow-primary/20 group-hover:scale-105 transition-transform">
              <Leaf size={18} />
            </div>
            <span className="font-display text-xl font-semibold tracking-tight text-foreground">
              Smart-Food <span className="text-primary">Link</span>
            </span>
          </Link>
        </div>

        <div className="flex-1 overflow-y-auto py-6 px-4 space-y-1">
          {NAV_ITEMS.map((item) => {
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200 font-medium text-sm",
                  isActive
                    ? "bg-primary/10 text-primary shadow-sm"
                    : "text-muted-foreground hover:bg-secondary hover:text-foreground"
                )}
              >
                <item.icon size={18} className={cn(isActive ? "text-primary" : "text-muted-foreground")} />
                {item.label}
              </Link>
            );
          })}
        </div>

        <div className="p-4 border-t border-border/40 bg-card/50 space-y-3">
          <ApiStatusIndicator />
          {user && (
            <button
              onClick={handleLogout}
              className="flex items-center gap-2 w-full px-3 py-2 rounded-lg text-sm font-medium text-muted-foreground hover:bg-destructive/10 hover:text-destructive transition-colors"
            >
              <LogOut size={16} />
              Sign Out
            </button>
          )}
        </div>
      </aside>

      {/* Mobile Header */}
      <div className="md:hidden fixed top-0 inset-x-0 h-16 bg-card/80 backdrop-blur-md border-b border-border/60 z-30 flex items-center justify-between px-4">
        <Link href="/" className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-primary to-emerald-500 flex items-center justify-center text-white shadow-sm">
            <Leaf size={14} />
          </div>
          <span className="font-display text-lg font-semibold tracking-tight">Smart-Food</span>
        </Link>
        <button
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          className="p-2 -mr-2 text-foreground hover:bg-secondary rounded-lg"
        >
          {mobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
        </button>
      </div>

      {/* Mobile Menu Overlay */}
      <AnimatePresence>
        {mobileMenuOpen && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="md:hidden fixed inset-0 z-20 bg-card pt-16"
          >
            <div className="p-4 space-y-2">
              {NAV_ITEMS.map((item) => {
                const isActive = pathname === item.href;
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={cn(
                      "flex items-center gap-3 px-4 py-4 rounded-xl transition-colors font-medium text-base",
                      isActive
                        ? "bg-primary/10 text-primary"
                        : "text-muted-foreground hover:bg-secondary"
                    )}
                  >
                    <item.icon size={20} />
                    {item.label}
                  </Link>
                );
              })}

              {user && (
                <button
                  onClick={handleLogout}
                  className="flex items-center gap-3 px-4 py-4 rounded-xl transition-colors font-medium text-base text-destructive hover:bg-destructive/10 w-full"
                >
                  <LogOut size={20} />
                  Sign Out
                </button>
              )}
            </div>

            <div className="absolute bottom-0 inset-x-0 p-4 border-t border-border">
              <ApiStatusIndicator />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
