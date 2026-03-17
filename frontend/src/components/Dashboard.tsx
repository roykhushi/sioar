"use client";

import {
  AlertTriangle, Handshake, Building2, Leaf,
  ArrowRight, TrendingUp, BrainCircuit, Activity
} from "lucide-react";
import Link from "next/link";
import { SectionCard } from "@/src/components/ui/SectionCard";
import { StatusBadge } from "@/src/components/ui/StatusBadge";
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip as RechartsTooltip, Legend } from "recharts";
import { motion } from "framer-motion";
import toast from "react-hot-toast";
import { useState, useEffect } from "react";
import axios from "axios";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8080";

interface DashboardMetrics {
  high_risk_count: number;
  donations_count: number;
  waste_prevented_kg: number;
  active_ngos_count: number;
  risk_distribution: {
    High: number;
    Medium: number;
    Low: number;
  };
}

interface Prediction {
  _id?: string;
  id?: string;
  item_name: string;
  category: string;
  risk_level: "High" | "Medium" | "Low";
  action: string;
  created_at?: string;
}

export default function Dashboard() {
  const [metrics, setMetrics] = useState<DashboardMetrics | null>(null);
  const [recentPredictions, setRecentPredictions] = useState<Prediction[]>([]);
  const [chartData, setChartData] = useState<any[]>([]);
  const [isTraining, setIsTraining] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [isError, setIsError] = useState(false);
  const [mounted, setMounted] = useState(false);

  // Set mounted flag to prevent hydration mismatch
  useEffect(() => {
    setMounted(true);
  }, []);

  // Fetch dashboard metrics
  useEffect(() => {
    const fetchDashboardData = async () => {
      setIsLoading(true);
      setIsError(false);
      try {
        // Fetch all metrics in one call
        const metricsRes = await axios.get(`${API_BASE_URL}/dashboard/metrics`);
        setMetrics(metricsRes.data);

        // Prepare chart data from risk distribution
        const chartData = [
          {
            name: "Low Risk",
            value: metricsRes.data.risk_distribution.Low,
            color: "hsl(142, 72%, 29%)"
          },
          {
            name: "Medium Risk",
            value: metricsRes.data.risk_distribution.Medium,
            color: "hsl(38, 92%, 50%)"
          },
          {
            name: "High Risk",
            value: metricsRes.data.risk_distribution.High,
            color: "hsl(0, 84%, 60%)"
          },
        ];
        setChartData(chartData);

        // Fetch recent predictions
        const predsRes = await axios.get(`${API_BASE_URL}/predictions?limit=5`);
        setRecentPredictions(predsRes.data || []);
      } catch (error) {
        console.error("Error fetching dashboard data:", error);
        setIsError(true);
        toast.error("Failed to load dashboard data");
      } finally {
        setIsLoading(false);
      }
    };

    fetchDashboardData();
  }, []);

  const handleTrainModel = async () => {
    toast.loading("Training model...", { id: "train" });
    setIsTraining(true);
    try {
      const res = await axios.post(`${API_BASE_URL}/train`, {});
      toast.success(res.data.message || "Model trained successfully!", { id: "train" });
      // Refresh dashboard data after training
      setTimeout(() => window.location.reload(), 1000);
    } catch {
      toast.error("Failed to train model", { id: "train" });
    } finally {
      setIsTraining(false);
    }
  };

  const containerVariants = {
    hidden: { opacity: 0 },
    show: { opacity: 1, transition: { staggerChildren: 0.1 } }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0, transition: { type: "spring", stiffness: 300, damping: 24 } }
  };

  // Format time ago
  const formatTimeAgo = (date: string) => {
    const now = new Date();
    const then = new Date(date);
    const seconds = Math.floor((now.getTime() - then.getTime()) / 1000);

    if (seconds < 60) return "Just now";
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
    return `${Math.floor(seconds / 86400)}d ago`;
  };

  return (
    <div className="space-y-6 lg:space-y-8 pb-10">

      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4">
        <div>
          <h1 className="text-3xl sm:text-4xl font-display text-foreground font-bold tracking-tight mb-1">
            Overview
          </h1>
          <p className="text-muted-foreground text-sm sm:text-base">
            AI-powered insights to minimize food waste and maximize impact.
          </p>
        </div>
        <div className="flex gap-3">
          <button
            onClick={handleTrainModel}
            disabled={isTraining}
            className="px-4 py-2 bg-secondary text-primary font-semibold rounded-xl hover:bg-primary/15 transition-colors flex items-center gap-2 text-sm disabled:opacity-50"
          >
            <Activity size={16} className={isTraining ? "animate-pulse" : ""} />
            {isTraining ? "Training..." : "Retrain AI"}
          </button>
          <Link
            href="/predict"
            className="px-4 py-2 bg-linear-to-r from-primary to-emerald-500 text-white font-semibold rounded-xl shadow-lg shadow-primary/25 hover:shadow-xl hover:shadow-primary/30 hover:-translate-y-0.5 transition-all duration-200 flex items-center gap-2 text-sm"
          >
            <BrainCircuit size={16} />
            New Prediction
          </Link>
        </div>
      </div>

      {/* Stats Row */}
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="show"
        className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4"
      >
        {/* Items at Risk */}
        <motion.div variants={itemVariants} className="bg-card rounded-2xl p-5 border border-border/60 organic-shadow relative overflow-hidden group">
          <div className="absolute left-0 top-0 bottom-0 w-1.5 bg-destructive transition-all duration-300 group-hover:w-2"></div>
          <div className="flex justify-between items-start mb-4">
            <div className="bg-destructive/10 p-2.5 rounded-xl">
              <AlertTriangle className="text-destructive" size={20} />
            </div>
            {(metrics?.high_risk_count || 0) > 0 && (
              <span className="flex items-center text-xs font-medium text-destructive bg-destructive/10 px-2 py-1 rounded-full">
                <TrendingUp size={12} className="mr-1" /> High Alert
              </span>
            )}
          </div>
          <div>
            <p className="text-muted-foreground text-sm font-medium mb-1">Items at Risk</p>
            <h3 className="text-3xl font-display font-bold text-foreground">
              {isLoading ? "..." : metrics?.high_risk_count || 0}
            </h3>
          </div>
        </motion.div>

        {/* Donations Matched */}
        <motion.div variants={itemVariants} className="bg-card rounded-2xl p-5 border border-border/60 organic-shadow relative overflow-hidden group">
          <div className="absolute left-0 top-0 bottom-0 w-1.5 bg-primary transition-all duration-300 group-hover:w-2"></div>
          <div className="flex justify-between items-start mb-4">
            <div className="bg-primary/10 p-2.5 rounded-xl">
              <Handshake className="text-primary" size={20} />
            </div>
          </div>
          <div>
            <p className="text-muted-foreground text-sm font-medium mb-1">Donations Matched</p>
            <h3 className="text-3xl font-display font-bold text-foreground">
              {isLoading ? "..." : metrics?.donations_count || 0}
            </h3>
          </div>
        </motion.div>

        {/* Active NGOs */}
        <motion.div variants={itemVariants} className="bg-card rounded-2xl p-5 border border-border/60 organic-shadow relative overflow-hidden group">
          <div className="absolute left-0 top-0 bottom-0 w-1.5 bg-emerald-500 transition-all duration-300 group-hover:w-2"></div>
          <div className="flex justify-between items-start mb-4">
            <div className="bg-emerald-500/10 p-2.5 rounded-xl">
              <Building2 className="text-emerald-600" size={20} />
            </div>
          </div>
          <div>
            <p className="text-muted-foreground text-sm font-medium mb-1">Active NGOs</p>
            <h3 className="text-3xl font-display font-bold text-foreground">
              {isLoading ? "..." : metrics?.active_ngos_count || 0}
            </h3>
          </div>
        </motion.div>

        {/* Waste Prevented */}
        <motion.div variants={itemVariants} className="bg-card rounded-2xl p-5 border border-border/60 organic-shadow relative overflow-hidden group">
          <div className="absolute left-0 top-0 bottom-0 w-1.5 bg-success transition-all duration-300 group-hover:w-2"></div>
          <div className="flex justify-between items-start mb-4">
            <div className="bg-success/10 p-2.5 rounded-xl">
              <Leaf className="text-success" size={20} />
            </div>
          </div>
          <div>
            <p className="text-muted-foreground text-sm font-medium mb-1">Waste Prevented</p>
            <h3 className="text-3xl font-display font-bold text-foreground">
              {isLoading ? "..." : Math.round(metrics?.waste_prevented_kg || 0)}
              <span className="text-lg text-muted-foreground font-sans font-medium"> kg</span>
            </h3>
          </div>
        </motion.div>
      </motion.div>

      {/* Main Content Grids */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 lg:gap-8">
        {/* Risk Distribution Chart */}
        <SectionCard title="Risk Distribution" delay={0.4} className="lg:col-span-1 min-h-[380px]">
          <div className="h-65 w-full mt-4">
            {isLoading || chartData.length === 0 ? (
              <div className="flex items-center justify-center h-full text-muted-foreground">
                Loading chart...
              </div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie data={chartData} cx="50%" cy="50%" innerRadius={65} outerRadius={90} paddingAngle={5} dataKey="value" stroke="none">
                    {chartData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <RechartsTooltip contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 8px 30px rgba(0,0,0,0.08)' }} itemStyle={{ color: '#0f172a', fontWeight: 600 }} />
                  <Legend verticalAlign="bottom" height={36} iconType="circle" />
                </PieChart>
              </ResponsiveContainer>
            )}
          </div>
        </SectionCard>

        {/* Recent Predictions Table */}
        <SectionCard
          title="Recent Predictions"
          className="lg:col-span-2"
          delay={0.5}
          action={
            <Link href="/predictions" className="text-sm font-medium text-primary hover:text-emerald-700 flex items-center gap-1 group">
              View All <ArrowRight size={14} className="group-hover:translate-x-1 transition-transform" />
            </Link>
          }
        >
          <div className="overflow-x-auto -mx-6 sm:mx-0">
            <table className="w-full text-sm text-left">
              <thead className="text-xs text-muted-foreground uppercase bg-secondary/30">
                <tr>
                  <th className="px-6 py-3 font-semibold rounded-tl-lg">Item</th>
                  <th className="px-6 py-3 font-semibold">Category</th>
                  <th className="px-6 py-3 font-semibold">Risk Level</th>
                  <th className="px-6 py-3 font-semibold">Action</th>
                  <th className="px-6 py-3 font-semibold rounded-tr-lg">Time</th>
                </tr>
              </thead>
              <tbody>
                {isLoading ? (
                  <tr>
                    <td colSpan={5} className="px-6 py-8 text-center text-muted-foreground">
                      Loading predictions...
                    </td>
                  </tr>
                ) : recentPredictions.length === 0 ? (
                  <tr>
                    <td colSpan={5} className="px-6 py-8 text-center text-muted-foreground">
                      No predictions yet. Make your first prediction!
                    </td>
                  </tr>
                ) : (
                  recentPredictions.map((row) => (
                    <tr key={row._id} className="border-b border-border/40 hover:bg-secondary/20 transition-colors last:border-0">
                      <td className="px-6 py-4 font-medium text-foreground whitespace-nowrap">{row.item_name}</td>
                      <td className="px-6 py-4 text-muted-foreground">{row.category}</td>
                      <td className="px-6 py-4"><StatusBadge level={row.risk_level} /></td>
                      <td className="px-6 py-4 text-foreground">{row.action}</td>
                      <td className="px-6 py-4 text-muted-foreground whitespace-nowrap">
                        {mounted && row.created_at ? formatTimeAgo(row.created_at) : "Recently"}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </SectionCard>
      </div>
    </div>
  );
}