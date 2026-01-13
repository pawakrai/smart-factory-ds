import Dashboard from "../components/Dashboard";

export default function Home() {
  return (
    <main style={{ padding: 24 }}>
      <h1>Daily Planning Dashboard (Prototype)</h1>
      <p>
        ตั้งค่า และกด Recalculate เพื่อเรียก backend /api/schedule แล้วดูผลลัพธ์
      </p>
      <Dashboard />
    </main>
  );
}



