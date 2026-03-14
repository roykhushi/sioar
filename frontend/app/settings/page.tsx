"use client";

import { useState, useEffect } from "react";
import { SectionCard } from "@/src/components/ui/SectionCard";
import { Settings, Server, Shield, CheckCircle2, XCircle } from "lucide-react";
import toast from "react-hot-toast";
import axios from "axios";

const DEFAULT_API_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8080";

export default function SettingsPage() {
  const [apiUrl, setApiUrl] = useState(DEFAULT_API_URL);
  const [isFetching, setIsFetching] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<"idle" | "ok" | "error">("idle");

  // ← Load from localStorage on mount
  useEffect(() => {
    const saved = localStorage.getItem("smart_food_api_url");
    if (saved) setApiUrl(saved);
  }, []);

  const handleSave = () => {
    localStorage.setItem("smart_food_api_url", apiUrl);
    toast.success("Settings saved! Reload the page to apply changes.");
  };

  // ← replaces useHealthCheck() + refetch()
  const testConnection = async () => {
    toast.loading("Testing connection...", { id: "test-conn" });
    setIsFetching(true);
    setConnectionStatus("idle");
    try {
      const res = await axios.get(`${apiUrl}/`);
      if (res.status === 200) {
        setConnectionStatus("ok");
        toast.success("Connection successful!", { id: "test-conn" });
      } else {
        setConnectionStatus("error");
        toast.error("Connection failed.", { id: "test-conn" });
      }
    } catch {
      setConnectionStatus("error");
      toast.error("Connection failed.", { id: "test-conn" });
    } finally {
      setIsFetching(false);
    }
  };

  return (
    <div className="max-w-4xl space-y-6 pb-10">
      <div>
        <h1 className="text-3xl font-display text-foreground font-bold tracking-tight mb-1 flex items-center gap-3">
          <div className="p-2 bg-secondary text-foreground rounded-xl">
            <Settings size={28} />
          </div>
          Settings
        </h1>
        <p className="text-muted-foreground">
          Configure application preferences and system integrations.
        </p>
      </div>

      <SectionCard title="API Configuration" className="max-w-2xl">
        <div className="space-y-6">

          <div className="flex items-start gap-4 p-4 rounded-xl bg-secondary/30 border border-border">
            <Server className="text-primary mt-0.5" />
            <div className="flex-1">
              <h4 className="font-semibold text-foreground text-sm">Server Connection</h4>
              <p className="text-xs text-muted-foreground mt-1 mb-3">
                Point the dashboard to your running backend API instance.
              </p>

              <div className="space-y-2 max-w-md">
                <label className="text-xs font-semibold text-foreground uppercase tracking-wider">Base URL</label>
                <div className="flex gap-2">
                  <input
                    type="url"
                    value={apiUrl}
                    onChange={(e) => {
                      setApiUrl(e.target.value);
                      setConnectionStatus("idle");
                    }}
                    className="flex-1 px-4 py-2 bg-background border border-border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary font-mono text-sm"
                    placeholder="e.g. http://localhost:8080"
                  />
                  <button
                    onClick={handleSave}
                    className="px-4 py-2 bg-foreground text-background font-medium rounded-lg hover:bg-foreground/90 transition-colors text-sm whitespace-nowrap"
                  >
                    Save
                  </button>
                </div>
              </div>

              <div className="mt-4 pt-4 border-t border-border/50">
                <div className="flex items-center gap-4">
                  <button
                    onClick={testConnection}
                    disabled={isFetching}
                    className="px-4 py-2 bg-secondary text-foreground font-medium rounded-lg hover:bg-border transition-colors text-sm disabled:opacity-50 disabled:pointer-events-none"
                  >
                    {isFetching ? "Testing..." : "Test Connection"}
                  </button>

                  {connectionStatus === "ok" && (
                    <span className="flex items-center text-sm font-medium text-success">
                      <CheckCircle2 size={16} className="mr-1.5" /> Healthy
                    </span>
                  )}
                  {connectionStatus === "error" && (
                    <span className="flex items-center text-sm font-medium text-destructive">
                      <XCircle size={16} className="mr-1.5" /> Offline
                    </span>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* ← Updated: removed Replit-specific auth note */}
          <div className="flex items-start gap-4 p-4 rounded-xl bg-secondary/30 border border-border">
            <Shield className="text-muted-foreground mt-0.5" />
            <div>
              <h4 className="font-semibold text-foreground text-sm">Authentication</h4>
              <p className="text-xs text-muted-foreground mt-1">
                API requests are unauthenticated. Secure your backend with an API key or auth middleware before deploying to production.
              </p>
            </div>
          </div>

        </div>
      </SectionCard>
    </div>
  );
}