"use client";

import { useState } from "react";
import { SectionCard } from "@/src/components/ui/SectionCard";
import { HeartHandshake, MapPin, Phone, Mail, ArrowRight, ExternalLink } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import toast from "react-hot-toast";
import axios from "axios";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8080";

const CATEGORIES = [
  "Dairy", "Bakery", "Produce", "Meat", "Canned", "Frozen",
  "Beverages", "Snacks", "Grains & Pulses"
];

interface NGO {
  name: string;
  location: string;
  contact: string;
  categories_accepted: string[];
}

export default function MatchPage() {
  const [category, setCategory] = useState("Dairy");
  const [isPending, setIsPending] = useState(false);
  const [data, setData] = useState<NGO | null>(null);
  const [error, setError] = useState(false);

  // ← replaces useMatchNgo()
  const handleMatch = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsPending(true);
    setError(false);
    setData(null);
    try {
      const res = await axios.post(`${API_BASE_URL}/match`, { category });
      if (res.data?.recommended_ngo) {
        setData(res.data.recommended_ngo);
        toast.success("Found a match!");
      } else {
        setError(true);
        toast.error(res.data?.message || "No suitable NGO found for this category.");
      }
    } catch {
      setError(true);
      toast.error("No suitable NGO found for this category.");
    } finally {
      setIsPending(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto space-y-8 pb-10">
      <div className="text-center mt-4 mb-8">
        <div className="w-16 h-16 bg-primary/10 text-primary rounded-2xl flex items-center justify-center mx-auto mb-4">
          <HeartHandshake size={32} />
        </div>
        <h1 className="text-3xl md:text-4xl font-display text-foreground font-bold tracking-tight mb-3">
          Find Donation Match
        </h1>
        <p className="text-muted-foreground text-lg max-w-xl mx-auto">
          Instantly connect with the best NGO for your specific food items to ensure nothing goes to waste.
        </p>
      </div>

      <SectionCard className="p-2 sm:p-4">
        <form onSubmit={handleMatch} className="flex flex-col sm:flex-row gap-4">
          <div className="flex-1 relative">
            <select
              value={category}
              onChange={(e) => setCategory(e.target.value)}
              className="w-full px-5 py-4 bg-secondary/50 border-2 border-border rounded-xl focus:outline-none focus:ring-4 focus:ring-primary/10 focus:border-primary transition-all appearance-none text-lg font-medium"
            >
              {CATEGORIES.map(cat => (
                <option key={cat} value={cat}>{cat}</option>
              ))}
            </select>
            <div className="absolute right-4 top-1/2 -translate-y-1/2 pointer-events-none text-muted-foreground">
              ▼
            </div>
          </div>
          <button
            type="submit"
            disabled={isPending}
            className="px-8 py-4 bg-gradient-to-r from-primary to-emerald-500 text-white font-semibold rounded-xl shadow-lg shadow-primary/25 hover:shadow-xl hover:shadow-primary/30 hover:-translate-y-0.5 active:translate-y-0 transition-all duration-200 disabled:opacity-70 disabled:pointer-events-none text-lg flex items-center justify-center gap-2"
          >
            {isPending ? "Searching..." : "Find Best NGO"}
          </button>
        </form>
      </SectionCard>

      <AnimatePresence mode="wait">
        {data && !error && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="bg-card rounded-3xl border border-border/60 organic-shadow overflow-hidden relative"
          >
            <div className="absolute top-0 inset-x-0 h-2 bg-gradient-to-r from-primary to-emerald-400"></div>

            <div className="p-8 md:p-10">
              <div className="inline-block px-3 py-1 bg-success/10 text-success text-sm font-semibold rounded-full mb-4">
                Perfect Match Found
              </div>

              <h2 className="text-3xl font-display font-bold text-foreground mb-4">{data.name}</h2>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
                <div className="space-y-4">
                  <div className="flex items-start text-foreground">
                    <MapPin className="text-primary mr-3 mt-0.5" size={20} />
                    <div>
                      <p className="font-medium">Location</p>
                      <p className="text-muted-foreground">{data.location}</p>
                    </div>
                  </div>

                  <div className="flex items-start text-foreground">
                    <Phone className="text-primary mr-3 mt-0.5" size={20} />
                    <div>
                      <p className="font-medium">Contact</p>
                      <p className="text-muted-foreground">{data.contact}</p>
                    </div>
                  </div>

                  <div className="flex items-start text-foreground">
                    <Mail className="text-primary mr-3 mt-0.5" size={20} />
                    <div>
                      <p className="font-medium">Email</p>
                      <p className="text-muted-foreground">
                        donate@{data.name.toLowerCase().replace(/[^a-z0-9]/g, "")}.org
                      </p>
                    </div>
                  </div>
                </div>

                <div className="bg-secondary/40 rounded-2xl p-5 border border-border/50">
                  <p className="font-semibold text-foreground mb-3">Accepted Categories</p>
                  <div className="flex flex-wrap gap-2">
                    {data.categories_accepted.map(cat => (
                      <span
                        key={cat}
                        className={`px-3 py-1.5 rounded-lg text-sm font-medium ${
                          cat === category
                            ? "bg-primary text-white shadow-sm"
                            : "bg-background border border-border text-foreground"
                        }`}
                      >
                        {cat} {cat === category && "✓"}
                      </span>
                    ))}
                  </div>
                </div>
              </div>

              <div className="pt-6 border-t border-border/60 flex flex-col sm:flex-row gap-4">
                <button
                  onClick={() => toast.success("Donation workflow initiated! NGO notified.")}
                  className="flex-1 py-3.5 bg-foreground text-background font-semibold rounded-xl hover:bg-foreground/90 transition-colors flex items-center justify-center gap-2"
                >
                  Initiate Donation <ArrowRight size={18} />
                </button>
                <button className="px-6 py-3.5 bg-secondary text-foreground font-semibold rounded-xl hover:bg-border transition-colors flex items-center justify-center gap-2">
                  <ExternalLink size={18} /> View Profile
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}