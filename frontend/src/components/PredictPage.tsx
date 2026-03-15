"use client";

import { useState } from "react";
import { SectionCard } from "@/src/components/ui/SectionCard";
import { StatusBadge } from "@/src/components/ui/StatusBadge";
// import { LoadingSpinner } from "@/src/components/ui/LoadingSpinner";
import { BrainCircuit, Beaker, Package, IndianRupee, Calendar, TrendingUp } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import toast from "react-hot-toast";
import axios from "axios";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8080";

const CATEGORIES = [
  "Dairy", "Bakery", "Produce", "Meat", "Canned", "Frozen",
  "Beverages", "Snacks", "Grains & Pulses", "Unknown"
];

interface PredictionResult {
  Risk_Level: "High" | "Medium" | "Low";
  Probability: number;
  Action: string;
}

interface FormData {
  price: string;
  quantity: string;
  avg_daily_sales: string;
  days_until_expiry: string;
  category: string;
}

export default function PredictPage() {
  const [formData, setFormData] = useState<FormData>({
    price: "",
    quantity: "",
    avg_daily_sales: "",
    days_until_expiry: "",
    category: "Dairy"
  });

  const [isPending, setIsPending] = useState(false);
  const [data, setData] = useState<PredictionResult | null>(null);

  // ← replaces usePredictRisk()
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!formData.price || !formData.quantity || !formData.avg_daily_sales || !formData.days_until_expiry) {
      toast.error("Please fill in all numerical fields");
      return;
    }

    setIsPending(true);
    try {
      const res = await axios.post(`${API_BASE_URL}/predict`, {
        price: Number(formData.price),
        quantity: Number(formData.quantity),
        avg_daily_sales: Number(formData.avg_daily_sales),
        days_until_expiry: Number(formData.days_until_expiry),
        category: formData.category
      });
      setData(res.data);
      toast.success("Prediction generated successfully!");
    } catch (err) {
      toast.error("Failed to generate prediction. Please check your API connection.");
      console.error(err);
    } finally {
      setIsPending(false);
    }
  };

  // ← replaces reset()
  const handleReset = () => {
    setData(null);
    setFormData({
      price: "",
      quantity: "",
      avg_daily_sales: "",
      days_until_expiry: "",
      category: "Dairy"
    });
  };

  return (
    <div className="space-y-6 pb-10">
      <div>
        <h1 className="text-3xl font-display text-foreground font-bold tracking-tight mb-1 flex items-center gap-3">
          <div className="p-2 bg-primary/10 text-primary rounded-xl">
            <BrainCircuit size={28} />
          </div>
          Risk Prediction
        </h1>
        <p className="text-muted-foreground">
          Enter inventory metrics below to assess spoilage risk and get AI-driven recommendations.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 lg:gap-8">

        {/* Form Panel */}
        <SectionCard className="lg:col-span-7 h-fit" title="Inventory Details">
          <form onSubmit={handleSubmit} className="space-y-5 mt-2">

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
              <div className="space-y-2">
                <label className="text-sm font-semibold text-foreground flex items-center gap-2">
                  <IndianRupee size={14} className="text-muted-foreground" /> Unit Price (₹)
                </label>
                <input
                  type="number"
                  step="0.01"
                  min="0"
                  value={formData.price}
                  onChange={(e) => setFormData({ ...formData, price: e.target.value })}
                  className="w-full px-4 py-3 bg-secondary/50 border border-border rounded-xl focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all font-mono"
                  placeholder="e.g. 150.00"
                />
              </div>

              <div className="space-y-2">
                <label className="text-sm font-semibold text-foreground flex items-center gap-2">
                  <Package size={14} className="text-muted-foreground" /> Current Quantity
                </label>
                <input
                  type="number"
                  min="0"
                  value={formData.quantity}
                  onChange={(e) => setFormData({ ...formData, quantity: e.target.value })}
                  className="w-full px-4 py-3 bg-secondary/50 border border-border rounded-xl focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all font-mono"
                  placeholder="e.g. 50"
                />
              </div>

              <div className="space-y-2">
                <label className="text-sm font-semibold text-foreground flex items-center gap-2">
                  <TrendingUp size={14} className="text-muted-foreground" /> Avg. Daily Sales
                </label>
                <input
                  type="number"
                  step="0.1"
                  min="0"
                  value={formData.avg_daily_sales}
                  onChange={(e) => setFormData({ ...formData, avg_daily_sales: e.target.value })}
                  className="w-full px-4 py-3 bg-secondary/50 border border-border rounded-xl focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all font-mono"
                  placeholder="e.g. 5.5"
                />
              </div>

              <div className="space-y-2">
                <label className="text-sm font-semibold text-foreground flex items-center gap-2">
                  <Calendar size={14} className="text-muted-foreground" /> Days to Expiry
                </label>
                <input
                  type="number"
                  min="0"
                  value={formData.days_until_expiry}
                  onChange={(e) => setFormData({ ...formData, days_until_expiry: e.target.value })}
                  className="w-full px-4 py-3 bg-secondary/50 border border-border rounded-xl focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all font-mono"
                  placeholder="e.g. 3"
                />
              </div>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-semibold text-foreground">Food Category</label>
              <select
                value={formData.category}
                onChange={(e) => setFormData({ ...formData, category: e.target.value })}
                className="w-full px-4 py-3 bg-secondary/50 border border-border rounded-xl focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all appearance-none"
              >
                {CATEGORIES.map(cat => (
                  <option key={cat} value={cat}>{cat}</option>
                ))}
              </select>
            </div>

            <div className="pt-4 flex gap-3">
              <button
                type="submit"
                disabled={isPending}
                className="flex-1 py-3.5 bg-gradient-to-r from-primary to-emerald-500 text-white font-semibold rounded-xl shadow-lg shadow-primary/25 hover:shadow-xl hover:shadow-primary/30 hover:-translate-y-0.5 active:translate-y-0 active:shadow-md transition-all duration-200 disabled:opacity-70 disabled:pointer-events-none"
              >
                {isPending ? "Analyzing..." : "Predict Spoilage Risk"}
              </button>
              {data && (
                <button
                  type="button"
                  onClick={handleReset}
                  className="px-6 py-3.5 bg-secondary text-foreground font-semibold rounded-xl hover:bg-border transition-colors"
                >
                  Reset
                </button>
              )}
            </div>
          </form>
        </SectionCard>

        {/* Result Panel */}
        <div className="lg:col-span-5 flex flex-col h-full">
          <SectionCard className="flex-1 flex flex-col bg-gradient-to-br from-card to-card/50" title="Analysis Result">
            <div className="flex-1 flex flex-col justify-center min-h-[300px]">
              <AnimatePresence mode="wait">
                {isPending ? (
                  <motion.div
                    key="loading"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="flex flex-col items-center justify-center space-y-4"
                  >
                    <div className="relative">
                      <div className="absolute inset-0 bg-primary/20 rounded-full blur-xl animate-pulse"></div>
                      <BrainCircuit size={48} className="text-primary animate-bounce relative z-10" />
                    </div>
                    <p className="text-primary font-medium">Running AI models...</p>
                  </motion.div>
                ) : data ? (
                  <motion.div
                    key="result"
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="space-y-6"
                  >
                    <div className="text-center p-6 rounded-2xl bg-secondary/30 border border-border/50">
                      <p className="text-sm font-medium text-muted-foreground mb-3 uppercase tracking-wider">Assessed Risk Level</p>
                      <div className="flex justify-center">
                        <StatusBadge level={data.Risk_Level} className="text-lg px-4 py-1.5" />
                      </div>

                      <div className="mt-8">
                        <div className="flex justify-between text-sm mb-2 font-medium">
                          <span className="text-muted-foreground">Probability</span>
                          <span className="text-foreground">{(data.Probability * 100).toFixed(1)}%</span>
                        </div>
                        <div className="h-2.5 w-full bg-secondary rounded-full overflow-hidden">
                          <motion.div
                            initial={{ width: 0 }}
                            animate={{ width: `${data.Probability * 100}%` }}
                            transition={{ duration: 1, ease: "easeOut" }}
                            className={`h-full rounded-full ${
                              data.Risk_Level === "High" ? "bg-destructive" :
                              data.Risk_Level === "Medium" ? "bg-warning" : "bg-success"
                            }`}
                          />
                        </div>
                      </div>
                    </div>

                    <div className="p-5 rounded-2xl bg-primary/5 border border-primary/20">
                      <h4 className="font-semibold text-primary mb-2 flex items-center gap-2">
                        <Beaker size={16} /> Recommended Action
                      </h4>
                      <p className="text-foreground text-sm leading-relaxed">{data.Action}</p>
                    </div>

                  </motion.div>
                ) : (
                  <motion.div
                    key="empty"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="flex flex-col items-center justify-center text-center text-muted-foreground"
                  >
                    <div className="w-16 h-16 rounded-2xl bg-secondary flex items-center justify-center mb-4">
                      <BrainCircuit size={28} className="text-muted-foreground/50" />
                    </div>
                    <p className="font-medium">Awaiting Data</p>
                    <p className="text-sm mt-1 max-w-[250px]">Fill out the inventory details and click predict to see the AI analysis.</p>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </SectionCard>
        </div>

      </div>
    </div>
  );
}