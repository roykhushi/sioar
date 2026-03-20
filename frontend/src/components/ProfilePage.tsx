"use client";

import { useAuth } from "@/src/components/AuthContext";
import { SectionCard } from "@/src/components/ui/SectionCard";
import { UserCircle, Mail, Calendar, LogOut } from "lucide-react";
import { useRouter } from "next/navigation";
import toast from "react-hot-toast";

export default function ProfilePage() {
  const { user, logout } = useAuth();
  const router = useRouter();

  const handleLogout = () => {
    logout();
    toast.success("Signed out successfully");
    router.push("/signin");
  };

  const formattedDate = user?.created_at
    ? new Date(user.created_at).toLocaleDateString("en-IN", {
        year: "numeric",
        month: "long",
        day: "numeric",
      })
    : "N/A";

  return (
    <div className="max-w-4xl space-y-6 pb-10">
      <div>
        <h1 className="text-3xl font-display text-foreground font-bold tracking-tight mb-1 flex items-center gap-3">
          <div className="p-2 bg-primary/10 text-primary rounded-xl">
            <UserCircle size={28} />
          </div>
          Profile
        </h1>
        <p className="text-muted-foreground">
          Your account details and preferences.
        </p>
      </div>

      <SectionCard title="Account Information" className="max-w-2xl">
        <div className="space-y-6">
          <div className="flex items-center gap-5">
            <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-primary to-emerald-500 flex items-center justify-center text-white text-2xl font-display font-bold shadow-lg shrink-0">
              {user?.username?.[0]?.toUpperCase() || "U"}
            </div>
            <div>
              <h2 className="text-xl font-display font-semibold text-foreground">
                {user?.username}
              </h2>
              <p className="text-muted-foreground text-sm">{user?.email}</p>
            </div>
          </div>

          <div className="space-y-4 pt-4 border-t border-border/40">
            <div className="flex items-center gap-3 text-sm">
              <UserCircle size={16} className="text-primary shrink-0" />
              <span className="text-muted-foreground font-medium w-28">Username</span>
              <span className="text-foreground">{user?.username}</span>
            </div>
            <div className="flex items-center gap-3 text-sm">
              <Mail size={16} className="text-primary shrink-0" />
              <span className="text-muted-foreground font-medium w-28">Email</span>
              <span className="text-foreground">{user?.email}</span>
            </div>
            <div className="flex items-center gap-3 text-sm">
              <Calendar size={16} className="text-primary shrink-0" />
              <span className="text-muted-foreground font-medium w-28">Joined</span>
              <span className="text-foreground">{formattedDate}</span>
            </div>
          </div>

          <div className="pt-4 border-t border-border/40">
            <button
              onClick={handleLogout}
              className="flex items-center gap-2 px-5 py-2.5 bg-destructive/10 text-destructive font-medium rounded-xl hover:bg-destructive/20 transition-colors text-sm"
            >
              <LogOut size={16} />
              Sign Out
            </button>
          </div>
        </div>
      </SectionCard>
    </div>
  );
}
