import React from "react";
import cloud from "./assets/image.png";
import "./App.css";

const mockData = {
  location: "Thung Khru District",
  current: {
    temp: 27,
    condition: "Cloudy",
    feelsLike: 28,
    humidity: 72,
  },
  forecast: [
    { day: "Today", risk: "Low" },
    { day: "Tomorrow", risk: "Medium" },
    { day: "Wed", risk: "Low" },
    { day: "Thu", risk: "High" },
    { day: "Fri", risk: "Low" },
    { day: "Sat", risk: "Medium" },
    { day: "Sun", risk: "Low" },
  ],
};


const getRiskIcon = (risk) => {
  switch (risk) {
    case "Low":
      return "🟢"; // ความเสี่ยงต่ำ
    case "Medium":
      return "🟡"; // ความเสี่ยงปานกลาง
    case "High":
      return "🔴"; // ความเสี่ยงสูง
    default:
      return "⚪";
  }
};

export default function App() {
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
      }}
    >
    
      <img src={cloud} className="cloud c1" />
      <img src={cloud} className="cloud c2" />
      <img src={cloud} className="cloud c3" />
      <img src={cloud} className="cloud-r r1" />
      <img src={cloud} className="cloud-r r2" />
      <img src={cloud} className="cloud-r r3" />

      {/* Main Weather Box */}
      <div
        style={{
          padding: "30px",
          backdropFilter: "blur(8px)",
          borderRadius: "2rem",
          backgroundColor: "rgba(255, 255, 255, 0.15)",
          textShadow: "2px 2px 6px rgba(0,0,0,0.8)",
          width: "90%",
          maxWidth: "600px",
          marginBottom: "30px",
        }}
      >
        <div style={{ fontSize: "1rem", textTransform: "uppercase", letterSpacing: "2px" }}>
          My Location
        </div>

        <h1 style={{ fontSize: "4rem", fontWeight: "700", margin: "0.5rem 0", textShadow: "2px 2px 8px rgba(0,0,0,0.9)" }}>
          {mockData.location}
        </h1>

        <div style={{ fontSize: "1.5rem", opacity: 0.95, textShadow: "1px 1px 5px rgba(0,0,0,0.9)" }}>
          {mockData.current.condition}
        </div>

        <div style={{ fontSize: "6rem", fontWeight: "800", margin: "1rem 0", textShadow: "3px 3px 10px rgba(0,0,0,0.9)" }}>
          {mockData.current.temp}°
        </div>

        <div style={{ fontSize: "1.2rem", opacity: 0.9, marginBottom: "1rem", textShadow: "1px 1px 6px rgba(0,0,0,0.9)" }}>
          Feels like {mockData.current.feelsLike}° • Humidity {mockData.current.humidity}%
        </div>
      </div>

      {/* Forecast Container รวมทั้งหมด */}
      <div
        style={{
          width: "90%",
          maxWidth: "600px",
          borderRadius: "1.5rem",
          backgroundColor: "rgba(79, 85, 107, 0.2)", 
          padding: "10px",
          boxShadow: "0 4px 8px rgba(0,0,0,0.3)",
        }}
      >
        {mockData.forecast.map((day, idx) => (
          <div
            key={idx}
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              padding: "12px 15px",
              borderBottom: idx < mockData.forecast.length - 1 ? "1px solid rgba(255,255,255,0.2)" : "none",
              color: "white",
              fontWeight: "bold",
              textShadow: "1px 1px 4px rgba(0, 0, 0, 0.65)",
              fontSize: "1.25rem",
            }}
          >
            <div style={{ flex: 1, textAlign: "left" }}>{day.day}</div>
            <div style={{ flex: 1, fontSize: "1.5rem", textAlign: "center" }}>{getRiskIcon(day.risk)}</div>
            <div style={{ flex: 1, textAlign: "right" }}>{day.risk} Risk</div>
          </div>
        ))}
      </div>
    </div>
  );
}
