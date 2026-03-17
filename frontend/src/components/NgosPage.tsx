"use client";

import { useState, useEffect } from "react";
import { Building2, MapPin, Phone, Search, Mail } from "lucide-react";
import { motion } from "framer-motion";
import axios from "axios";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8080";

interface NGO {
  id: number;
  name: string;
  location: string;
  contact: string;
  categories_accepted: string[];
}

export default function NgosPage() {
  const [ngos, setNgos] = useState<NGO[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isError, setIsError] = useState(false);
  const [searchTerm, setSearchTerm] = useState("");

  // ← replaces useListNgos()
  useEffect(() => {
    axios.get(`${API_BASE_URL}/ngos`)
      .then(res => {
        setNgos(res.data);
        setIsLoading(false);
      })
      .catch(() => {
        setIsError(true);
        setIsLoading(false);
      });
  }, []);

  const filteredNgos = ngos.filter(ngo =>
    ngo.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    (ngo.categories_accepted && ngo.categories_accepted.some(cat => cat.toLowerCase().includes(searchTerm.toLowerCase()))) ||
    ngo.location.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="space-y-6 pb-10">

      {/* Header & Search */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-6">
        <div>
          <h1 className="text-3xl font-display text-foreground font-bold tracking-tight mb-1 flex items-center gap-3">
            <div className="p-2 bg-emerald-500/10 text-emerald-600 rounded-xl">
              <Building2 size={28} />
            </div>
            NGO Directory
          </h1>
          <p className="text-muted-foreground">
            Connect with verified non-profits accepting food donations in your area.
          </p>
        </div>

        <div className="relative w-full md:w-80">
          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
            <Search size={18} className="text-muted-foreground" />
          </div>
          <input
            type="text"
            placeholder="Search NGOs or categories..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-10 pr-4 py-2.5 bg-card border border-border/80 rounded-xl focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all shadow-sm"
          />
        </div>
      </div>

      {/* States & Grid */}
      {isLoading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 pt-4">
          {[1, 2, 3, 4, 5, 6].map(i => (
            <div key={i} className="bg-card rounded-2xl p-6 border border-border/60 h-64 animate-pulse">
              <div className="h-6 w-2/3 bg-secondary rounded mb-4"></div>
              <div className="h-4 w-1/3 bg-secondary rounded mb-6"></div>
              <div className="flex gap-2 mb-6">
                <div className="h-6 w-16 bg-secondary rounded-full"></div>
                <div className="h-6 w-20 bg-secondary rounded-full"></div>
              </div>
              <div className="space-y-3 mt-auto">
                <div className="h-4 w-full bg-secondary rounded"></div>
                <div className="h-4 w-3/4 bg-secondary rounded"></div>
              </div>
            </div>
          ))}
        </div>
      ) : isError ? (
        <div className="text-center py-20 bg-card rounded-3xl border border-dashed border-border">
          <Building2 size={48} className="mx-auto text-muted-foreground/30 mb-4" />
          <h3 className="text-lg font-semibold text-foreground">Failed to load directory</h3>
          <p className="text-muted-foreground mt-1">Please check your API connection in settings.</p>
        </div>
      ) : filteredNgos.length === 0 ? (
        <div className="text-center py-20 bg-card rounded-3xl border border-dashed border-border">
          <Search size={48} className="mx-auto text-muted-foreground/30 mb-4" />
          <h3 className="text-lg font-semibold text-foreground">No NGOs found</h3>
          <p className="text-muted-foreground mt-1">Try adjusting your search terms.</p>
        </div>
      ) : (
        <motion.div
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
          initial="hidden"
          animate="show"
          variants={{
            hidden: { opacity: 0 },
            show: { opacity: 1, transition: { staggerChildren: 0.05 } }
          }}
        >
          {filteredNgos.map((ngo) => (
            <motion.div
              key={ngo.id}
              variants={{
                hidden: { opacity: 0, y: 20 },
                show: { opacity: 1, y: 0 }
              }}
              className="bg-card rounded-2xl border border-border/60 organic-shadow overflow-hidden flex flex-col group hover:border-emerald-500/30 transition-colors"
            >
              <div className="p-6 flex-1 flex flex-col">
                <div className="flex justify-between items-start mb-2">
                  <h3 className="text-xl font-display font-semibold text-foreground group-hover:text-primary transition-colors">
                    {ngo.name}
                  </h3>
                </div>

                <div className="flex items-center text-muted-foreground text-sm mb-5">
                  <MapPin size={14} className="mr-1.5" /> {ngo.location}
                </div>

                <div className="mb-6">
                  <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">Accepts</p>
                  <div className="flex flex-wrap gap-2">
                    {ngo.categories_accepted && ngo.categories_accepted.length > 0 ? (
                      ngo.categories_accepted.map(cat => (
                        <span key={cat} className="px-2.5 py-1 bg-secondary text-secondary-foreground text-xs font-medium rounded-md border border-border/50">
                          {cat}
                        </span>
                      ))
                    ) : (
                      <span className="text-xs text-muted-foreground">No categories specified</span>
                    )}
                  </div>
                </div>

                <div className="mt-auto pt-5 border-t border-border/40 space-y-2">
                  <div className="flex items-center text-sm text-foreground">
                    <Phone size={14} className="text-primary mr-2" />
                    {ngo.contact}
                  </div>
                  <div className="flex items-center text-sm text-foreground">
                    <Mail size={14} className="text-primary mr-2" />
                    contact@{ngo.name.toLowerCase().replace(/[^a-z0-9]/g, "")}.org
                  </div>
                </div>
              </div>
            </motion.div>
          ))}
        </motion.div>
      )}
    </div>
  );
}