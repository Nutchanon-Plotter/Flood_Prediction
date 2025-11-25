import React, { useEffect, useState } from "react";
import cloud from "./assets/image.png";
import "./App.css";

function getDayLabel(dateStr) {
  const date = new Date(dateStr);
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const diffDays = Math.round((date - today) / (1000 * 60 * 60 * 24));

  if (diffDays === 0) return "Today";
  if (diffDays === 1) return "Tomorrow";
  return date.toLocaleDateString("en-US", { weekday: "short" });
}

function formatFullDate(dateStr) {
  const date = new Date(dateStr);
  return date.toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
}

const getRiskIcon = (risk) => {
  switch (risk) {
    case "Low": return "💧";
    case "Medium": return "🌊";
    case "High": return "🚨🌊";
    default: return "⚪";
  }
};

export default function App() {
  const [forecast, setForecast] = useState([]);
  const [todayData, setTodayData] = useState(null);

  useEffect(() => {
    fetch("/prediction_results.json")
      .then((res) => res.json())
      .then((data) => {
        const sorted = [...data].sort(
          (a, b) => new Date(a.date) - new Date(b.date)
        );

        const firstDay = sorted[0];
        setTodayData({
          ...firstDay,
          fullDate: formatFullDate(firstDay.date),
        });

        const formatted = sorted.map((item) => ({
          ...item,
          dayLabel: getDayLabel(item.date),
          fullDate: formatFullDate(item.date),
        }));
        setForecast(formatted);
      })
      .catch((err) => console.error("Load JSON error:", err));
  }, []);

  if (!todayData) return <div style={{ color: "white" }}>Loading...</div>;

  return (
    <div
      style={{
        minHeight: "100vh",
        width: "100vw",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        backgroundImage:
          "linear-gradient(to bottom, #90c9ff 0%, #79b6ecff 50%, #ffffff 100%)",
        backgroundSize: "cover",
        backgroundAttachment: "fixed",
        backgroundPosition: "center",
        color: "white",
        textAlign: "center",
        overflowY: "auto",
        overflowX: "hidden",
        position: "relative",
        paddingTop: "5vh",
        paddingBottom: "50px",
        fontFamily: "'Montserrat', 'Roboto', sans-serif",
      }}
    >
      {/* Clouds Background */}
      <img src={cloud} className="cloud c1" alt="" />
      <img src={cloud} className="cloud c2" alt="" />
      <img src={cloud} className="cloud c3" alt="" />
      <img src={cloud} className="cloud-r r1" alt="" />
      <img src={cloud} className="cloud-r r2" alt="" />
      <img src={cloud} className="cloud-r r3" alt="" />

      {/* Today Box */}
      <div
        style={{
          padding: "30px",
          backdropFilter: "blur(8px)",
          borderRadius: "2rem",
          backgroundColor: "rgba(255, 255, 255, 0.15)",
          width: "90%",
          maxWidth: "600px",
          marginBottom: "30px",
          textShadow: "1px 1px 4px rgba(0,0,0,0.7)",
        }}
      >
        {/* --- 1. ส่วน Location --- */}
        <div style={{ 
            fontSize: "1.2rem", 
            opacity: 0.9, 
            fontWeight: "600", 
            textTransform: "uppercase",
            letterSpacing: "1px",
            marginBottom: "5px"
        }}>
          📍 {todayData.location || "Chao Phraya Dam"} 
        </div>

        <div style={{ fontSize: "1.5rem", opacity: 0.85, fontWeight: "500" }}>Today</div>
        
        <h1 style={{ fontSize: "3rem", fontWeight: "700", margin: "10px 0" }}>
          {todayData.fullDate}
        </h1>
        
        <div style={{ fontSize: "1.8rem", fontWeight: "500" }}>
          {todayData.status_text}
        </div>
        
        <div style={{ fontSize: "5rem", margin: "1rem 0" }}>
          {getRiskIcon(todayData.risk_level)}
        </div>

        {/* --- 2. ส่วน Flood Probability --- */}
        <div style={{ fontSize: "1.3rem", fontWeight: "500" }}>
          Flood Probability : {todayData.flood_probability.toFixed(2)} %
        </div>
        
        <div style={{ fontWeight: "600", marginTop: "5px", fontSize: "1.1rem" }}>
          Risk Level: {todayData.risk_level}
        </div>
      </div>

      {/* Forecast List */}
      <div
        style={{
          width: "90%",
          maxWidth: "600px",
          backgroundColor: "rgba(79,85,107,0.2)",
          borderRadius: "1.5rem",
          padding: "10px",
        }}
      >
        {forecast.map((day, idx) => (
          <div
            key={idx}
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              padding: "12px 15px",
              borderBottom:
                idx < forecast.length - 1
                  ? "1px solid rgba(255,255,255,0.2)"
                  : "none",
              color: "white",
              fontWeight: "bold",
              fontSize: "1.1rem", // ปรับขนาดตัวอักษรให้พอดีมือถือ
            }}
          >
            <div style={{ flex: 2, textAlign: "left" }}>
                <div>{day.fullDate}</div>
                <div style={{ fontSize: "0.8rem", opacity: 0.8, fontWeight: "normal" }}>
                    Prob: {day.flood_probability.toFixed(1)}%
                </div>
            </div>
            
            <div style={{ flex: 1, textAlign: "center", fontSize: "1.5rem" }}>
              {getRiskIcon(day.risk_level)}
            </div>
            
            <div style={{ flex: 2, textAlign: "right" }}>
                {day.risk_level} Risk
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}