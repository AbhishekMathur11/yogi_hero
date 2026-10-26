async function loadState() {
  const res = await fetch("/api/state");
  const state = await res.json();
  document.getElementById("camera-tag").textContent = state.camera;
  document.getElementById("phase-tag").textContent = state.phase;
  document.getElementById("asana-name").textContent = state.asana;
  document.getElementById("score-value").textContent =
    state.score == null ? "—" : Math.round(state.score * 100) + "%";
  document.getElementById("mdm-prompt").textContent = state.clip
    ? state.clip.prompt
    : "No MDM clip on disk";

  const list = document.getElementById("sequence");
  list.innerHTML = "";
  state.sequence.forEach((name) => {
    const item = document.createElement("li");
    const btn = document.createElement("button");
    btn.textContent = name;
    if (name === state.asana) btn.classList.add("active");
    btn.addEventListener("click", async () => {
      await fetch("/api/asana/" + encodeURIComponent(name), { method: "POST" });
      reloadMdm();
      loadState();
    });
    item.appendChild(btn);
    list.appendChild(item);
  });
}

function reloadMdm() {
  const video = document.getElementById("mdm");
  video.src = "/stream/mdm?t=" + Date.now();
  video.play().catch(() => {});
}

loadState();
reloadMdm();
setInterval(loadState, 2000);
