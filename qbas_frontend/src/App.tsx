import { NavLink, Route, Routes } from "react-router-dom";
import { ClipboardList, Fingerprint, Gauge, ScanFace, Sparkles } from "lucide-react";

import { Authenticate } from "./pages/Authenticate";
import { Audit } from "./pages/Audit";
import { Dashboard } from "./pages/Dashboard";
import { Enroll } from "./pages/Enroll";
import { Explainability } from "./pages/Explainability";
import { EnvironmentStatus } from "./components/EnvironmentStatus";

const navItems = [
  { to: "/", label: "Overview", icon: Gauge },
  { to: "/enroll", label: "Enroll identity", icon: Fingerprint },
  { to: "/authenticate", label: "Verify identity", icon: ScanFace },
  { to: "/audit", label: "Audit trail", icon: ClipboardList },
  { to: "/explainability", label: "Explainability", icon: Sparkles },
];

export default function App() {
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">I</div>
          <div>
            <strong>IndraNet</strong>
            <span>Identity assurance</span>
          </div>
        </div>
        <EnvironmentStatus />
        <nav>
          {navItems.map((item) => {
            const Icon = item.icon;

            return (
              <NavLink key={item.to} to={item.to} end={item.to === "/"}>
                <Icon size={18} />
                <span>{item.label}</span>
              </NavLink>
            );
          })}
        </nav>
      </aside>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/enroll" element={<Enroll />} />
        <Route path="/authenticate" element={<Authenticate />} />
        <Route path="/audit" element={<Audit />} />
        <Route path="/explainability" element={<Explainability />} />
      </Routes>
    </div>
  );
}
