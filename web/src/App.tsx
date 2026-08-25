import { Navigate, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { CommandCentre } from "./pages/CommandCentre";
import { Transfers } from "./pages/Transfers";
import { Captaincy } from "./pages/Captaincy";
import { Players } from "./pages/Players";
import { PlayerDetail } from "./pages/PlayerDetail";
import { Wildcard } from "./pages/Wildcard";
import { Fixtures } from "./pages/Fixtures";
import { Live } from "./pages/Live";
import { Settings } from "./pages/Settings";

export function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<CommandCentre />} />
        <Route path="transfers" element={<Transfers />} />
        <Route path="captaincy" element={<Captaincy />} />
        <Route path="players" element={<Players />} />
        <Route path="players/:id" element={<PlayerDetail />} />
        <Route path="wildcard" element={<Wildcard />} />
        <Route path="drafts" element={<Navigate to="/wildcard" replace />} />
        <Route path="fixtures" element={<Fixtures />} />
        <Route path="live" element={<Live />} />
        <Route path="settings" element={<Settings />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
