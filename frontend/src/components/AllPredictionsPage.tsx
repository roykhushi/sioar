"use client";

import { useState, useEffect } from "react";
import { ChevronLeft, ChevronRight, Search } from "lucide-react";
import { SectionCard } from "@/src/components/ui/SectionCard";
import { StatusBadge } from "@/src/components/ui/StatusBadge";
import axios from "axios";
import toast from "react-hot-toast";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8080";

interface Prediction {
  _id?: string;
  id?: string;
  item_name: string;
  price: number;
  quantity: number;
  category: string;
  risk_level: "High" | "Medium" | "Low";
  probability: number;
  action: string;
  created_at: string;
}

const ITEMS_PER_PAGE = 10;

export default function AllPredictionsPage() {
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [filteredPredictions, setFilteredPredictions] = useState<Prediction[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isError, setIsError] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const [searchTerm, setSearchTerm] = useState("");
  const [riskFilter, setRiskFilter] = useState<"all" | "High" | "Medium" | "Low">("all");
  const [mounted, setMounted] = useState(false);

  // Set mounted flag to prevent hydration mismatch
  useEffect(() => {
    setMounted(true);
  }, []);

  // Fetch all predictions
  useEffect(() => {
    const fetchPredictions = async () => {
      setIsLoading(true);
      setIsError(false);
      try {
        const res = await axios.get(`${API_BASE_URL}/predictions?limit=500`);
        setPredictions(res.data || []);
        setFilteredPredictions(res.data || []);
      } catch (error) {
        console.error("Error fetching predictions:", error);
        setIsError(true);
        toast.error("Failed to load predictions");
      } finally {
        setIsLoading(false);
      }
    };

    fetchPredictions();
  }, []);

  // Filter predictions
  useEffect(() => {
    let filtered = predictions;

    // Filter by risk level
    if (riskFilter !== "all") {
      filtered = filtered.filter(p => p.risk_level === riskFilter);
    }

    // Filter by search term
    if (searchTerm) {
      filtered = filtered.filter(p =>
        p.item_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        p.category.toLowerCase().includes(searchTerm.toLowerCase()) ||
        p.action.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    setFilteredPredictions(filtered);
    setCurrentPage(1); // Reset to first page
  }, [predictions, searchTerm, riskFilter]);

  // Pagination
  const totalPages = Math.ceil(filteredPredictions.length / ITEMS_PER_PAGE);
  const startIndex = (currentPage - 1) * ITEMS_PER_PAGE;
  const paginatedPredictions = filteredPredictions.slice(startIndex, startIndex + ITEMS_PER_PAGE);

  const formatTimeAgo = (date: string) => {
    const now = new Date();
    const then = new Date(date);
    const seconds = Math.floor((now.getTime() - then.getTime()) / 1000);

    if (seconds < 60) return "Just now";
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
    return `${Math.floor(seconds / 86400)}d ago`;
  };

  const formatPrice = (price: number) => `₹${price.toFixed(2)}`;

  return (
    <div className="space-y-6 pb-10">
      {/* Header */}
      <div>
        <h1 className="text-3xl sm:text-4xl font-display text-foreground font-bold tracking-tight mb-1">
          All Predictions
        </h1>
        <p className="text-muted-foreground text-sm sm:text-base">
          View and manage all historical food spoilage predictions from your inventory.
        </p>
      </div>

      {/* Filters */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {/* Search */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" size={18} />
          <input
            type="text"
            placeholder="Search by item, category, or action..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-10 pr-4 py-2.5 bg-secondary/50 border border-border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all"
          />
        </div>

        {/* Risk Filter */}
        <select
          value={riskFilter}
          onChange={(e) => setRiskFilter(e.target.value as any)}
          className="px-4 py-2.5 bg-secondary/50 border border-border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all appearance-none"
        >
          <option value="all">All Risk Levels</option>
          <option value="High">High Risk</option>
          <option value="Medium">Medium Risk</option>
          <option value="Low">Low Risk</option>
        </select>
      </div>

      {/* Results Info */}
      <div className="text-sm text-muted-foreground">
        Showing {startIndex + 1}-{Math.min(startIndex + ITEMS_PER_PAGE, filteredPredictions.length)} of {filteredPredictions.length} predictions
      </div>

      {/* Predictions Table */}
      <SectionCard title="Predictions" className="min-h-[500px]">
        <div className="overflow-x-auto -mx-6 sm:mx-0">
          <table className="w-full text-sm text-left">
            <thead className="text-xs text-muted-foreground uppercase bg-secondary/30 sticky top-0">
              <tr>
                <th className="px-6 py-3 font-semibold rounded-tl-lg">Item</th>
                <th className="px-6 py-3 font-semibold">Price</th>
                <th className="px-6 py-3 font-semibold">Qty</th>
                <th className="px-6 py-3 font-semibold">Category</th>
                <th className="px-6 py-3 font-semibold">Risk</th>
                <th className="px-6 py-3 font-semibold">Probability</th>
                <th className="px-6 py-3 font-semibold">Action</th>
                <th className="px-6 py-3 font-semibold rounded-tr-lg">Time</th>
              </tr>
            </thead>
            <tbody>
              {isLoading ? (
                <tr>
                  <td colSpan={8} className="px-6 py-8 text-center text-muted-foreground">
                    Loading predictions...
                  </td>
                </tr>
              ) : isError ? (
                <tr>
                  <td colSpan={8} className="px-6 py-8 text-center text-destructive">
                    Error loading predictions
                  </td>
                </tr>
              ) : paginatedPredictions.length === 0 ? (
                <tr>
                  <td colSpan={8} className="px-6 py-8 text-center text-muted-foreground">
                    No predictions found
                  </td>
                </tr>
              ) : (
                paginatedPredictions.map((pred) => (
                  <tr
                    key={pred._id}
                    className="border-b border-border/40 hover:bg-secondary/20 transition-colors last:border-0"
                  >
                    <td className="px-6 py-4 font-medium text-foreground whitespace-nowrap">
                      {pred.item_name}
                    </td>
                    <td className="px-6 py-4 text-muted-foreground whitespace-nowrap">
                      {formatPrice(pred.price)}
                    </td>
                    <td className="px-6 py-4 text-muted-foreground">{pred.quantity}</td>
                    <td className="px-6 py-4 text-muted-foreground">{pred.category}</td>
                    <td className="px-6 py-4">
                      <StatusBadge level={pred.risk_level} />
                    </td>
                    <td className="px-6 py-4 text-muted-foreground">
                      {(pred.probability * 100).toFixed(0)}%
                    </td>
                    <td className="px-6 py-4 text-foreground text-xs">
                      <span className="px-2.5 py-1 bg-secondary rounded-md whitespace-nowrap inline-block">
                        {pred.action}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-muted-foreground whitespace-nowrap">
                      {mounted ? formatTimeAgo(pred.created_at) : "Loading..."}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        {!isLoading && filteredPredictions.length > 0 && (
          <div className="mt-6 flex items-center justify-between border-t border-border/40 pt-4">
            <button
              onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
              disabled={currentPage === 1}
              className="flex items-center gap-2 px-4 py-2 bg-secondary text-foreground rounded-lg hover:bg-border disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <ChevronLeft size={18} />
              Previous
            </button>

            <div className="text-sm text-muted-foreground">
              Page {currentPage} of {totalPages}
            </div>

            <button
              onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
              disabled={currentPage === totalPages}
              className="flex items-center gap-2 px-4 py-2 bg-secondary text-foreground rounded-lg hover:bg-border disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              Next
              <ChevronRight size={18} />
            </button>
          </div>
        )}
      </SectionCard>
    </div>
  );
}
