const btn = document.getElementById("ping");
const out = document.getElementById("out");

btn.addEventListener("click", async () => {
  out.textContent = "Chiamo http://localhost:8000/health…";
  try {
    const res = await fetch("http://localhost:8000/health");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const json = await res.json();
    out.textContent = JSON.stringify(json, null, 2);
    out.className = "ok";
  } catch (err) {
    out.textContent = `Errore: ${err.message}`;
    out.className = "err";
  }
});
