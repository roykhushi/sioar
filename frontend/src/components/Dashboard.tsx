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

const CHART_DATA = [
  { name: "Low Risk", value: 450, color: "hsl(142, 72%, 29%)" },
  { name: "Medium Risk", value: 210, color: "hsl(38, 92%, 50%)" },
  { name: "High Risk", value: 85, color: "hsl(0, 84%, 60%)" },
];

const RECENT_PREDICTIONS = [
  { id: 1, item: "Organic Milk (1L)", category: "Dairy", risk: "High", action: "Donate immediately", time: "10 mins ago" },
  { id: 2, item: "Whole Wheat Bread", category: "Bakery", risk: "Medium", action: "Apply 20% discount", time: "1 hr ago" },
  { id: 3, item: "Gala Apples", category: "Produce", risk: "Low", action: "Regular sales", time: "2 hrs ago" },
  { id: 4, item: "Greek Yogurt", category: "Dairy", risk: "High", action: "Donate immediately", time: "3 hrs ago" },
  { id: 5, item: "Chicken Breast", category: "Meat", risk: "Medium", action: "Apply 30% discount", time: "5 hrs ago" },
];

export default function Dashboard() {
  const [ngos, setNgos] = useState<any[]>([]);
  const [isTraining, setIsTraining] = useState(false);

  useEffect(() => {
    axios.get(`${API_BASE_URL}/ngos`)
      .then(res => setNgos(res.data))
      .catch(() => {});
  }, []);

  const handleTrainModel = async () => {
    toast.loading("Training model...", { id: "train" });
    setIsTraining(true);
    try {
      const res = await axios.post(`${API_BASE_URL}/train`, {});
      toast.success(res.data.message || "Model trained successfully!", { id: "train" });
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
        <motion.div variants={itemVariants} className="bg-card rounded-2xl p-5 border border-border/60 organic-shadow relative overflow-hidden group">
          <div className="absolute left-0 top-0 bottom-0 w-1.5 bg-destructive transition-all duration-300 group-hover:w-2"></div>
          <div className="flex justify-between items-start mb-4">
            <div className="bg-destructive/10 p-2.5 rounded-xl">
              <AlertTriangle className="text-destructive" size={20} />
            </div>
            <span className="flex items-center text-xs font-medium text-destructive bg-destructive/10 px-2 py-1 rounded-full">
              <TrendingUp size={12} className="mr-1" /> +12%
            </span>
          </div>
          <div>
            <p className="text-muted-foreground text-sm font-medium mb-1">Items at Risk</p>
            <h3 className="text-3xl font-display font-bold text-foreground">85</h3>
          </div>
        </motion.div>

        <motion.div variants={itemVariants} className="bg-card rounded-2xl p-5 border border-border/60 organic-shadow relative overflow-hidden group">
          <div className="absolute left-0 top-0 bottom-0 w-1.5 bg-primary transition-all duration-300 group-hover:w-2"></div>
          <div className="flex justify-between items-start mb-4">
            <div className="bg-primary/10 p-2.5 rounded-xl">
              <Handshake className="text-primary" size={20} />
            </div>
          </div>
          <div>
            <p className="text-muted-foreground text-sm font-medium mb-1">Donations Matched</p>
            <h3 className="text-3xl font-display font-bold text-foreground">142</h3>
          </div>
        </motion.div>

        <motion.div variants={itemVariants} className="bg-card rounded-2xl p-5 border border-border/60 organic-shadow relative overflow-hidden group">
          <div className="absolute left-0 top-0 bottom-0 w-1.5 bg-emerald-500 transition-all duration-300 group-hover:w-2"></div>
          <div className="flex justify-between items-start mb-4">
            <div className="bg-emerald-500/10 p-2.5 rounded-xl">
              <Building2 className="text-emerald-600" size={20} />
            </div>
          </div>
          <div>
            <p className="text-muted-foreground text-sm font-medium mb-1">Active NGOs</p>
            <h3 className="text-3xl font-display font-bold text-foreground">{ngos?.length || "-"}</h3>
          </div>
        </motion.div>

        <motion.div variants={itemVariants} className="bg-card rounded-2xl p-5 border border-border/60 organic-shadow relative overflow-hidden group">
          <div className="absolute left-0 top-0 bottom-0 w-1.5 bg-success transition-all duration-300 group-hover:w-2"></div>
          <div className="flex justify-between items-start mb-4">
            <div className="bg-success/10 p-2.5 rounded-xl">
              <Leaf className="text-success" size={20} />
            </div>
          </div>
          <div>
            <p className="text-muted-foreground text-sm font-medium mb-1">Waste Prevented</p>
            <h3 className="text-3xl font-display font-bold text-foreground">1,204 <span className="text-lg text-muted-foreground font-sans font-medium">kg</span></h3>
          </div>
        </motion.div>
      </motion.div>

      {/* Main Content Grids */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 lg:gap-8">
        <SectionCard title="Risk Distribution" delay={0.4} className="lg:col-span-1 min-h-[380px]">
          <div className="h-65 w-full mt-4">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={CHART_DATA} cx="50%" cy="50%" innerRadius={65} outerRadius={90} paddingAngle={5} dataKey="value" stroke="none">
                  {CHART_DATA.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <RechartsTooltip contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 8px 30px rgba(0,0,0,0.08)' }} itemStyle={{ color: '#0f172a', fontWeight: 600 }} />
                <Legend verticalAlign="bottom" height={36} iconType="circle" />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </SectionCard>

        <SectionCard
          title="Recent Predictions"
          className="lg:col-span-2"
          delay={0.5}
          action={
            <Link href="/predict" className="text-sm font-medium text-primary hover:text-emerald-700 flex items-center gap-1 group">
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
                {RECENT_PREDICTIONS.map((row) => (
                  <tr key={row.id} className="border-b border-border/40 hover:bg-secondary/20 transition-colors last:border-0">
                    <td className="px-6 py-4 font-medium text-foreground whitespace-nowrap">{row.item}</td>
                    <td className="px-6 py-4 text-muted-foreground">{row.category}</td>
                    <td className="px-6 py-4"><StatusBadge level={row.risk as any} /></td>
                    <td className="px-6 py-4 text-foreground">{row.action}</td>
                    <td className="px-6 py-4 text-muted-foreground whitespace-nowrap">{row.time}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </SectionCard>
      </div>
    </div>
  );
}