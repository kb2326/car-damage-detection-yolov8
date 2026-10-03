// The adjuster's claim page: follow a running claim, resume a stopped one, record a review.
(() => {
  const data = JSON.parse(document.getElementById("claim-data").textContent);

  if (data.status === "running") {
    let last = null;
    const tick = async () => {
      const response = await fetch(`/api/claims/${data.id}`);
      if (!response.ok) return;
      const claim = await response.json();
      if (claim.status !== "running") { location.reload(); return; }
      const now = JSON.stringify(claim.stages);
      if (last !== null && now !== last) { location.reload(); return; }
      last = now;
      setTimeout(tick, data.poll * 1000);
    };
    setTimeout(tick, data.poll * 1000);
  }

  const resume = document.getElementById("resume");
  if (resume) {
    resume.addEventListener("click", async () => {
      resume.disabled = true;
      await fetch(`/api/claims/${data.id}/resume`, { method: "POST" });
      location.reload();
    });
  }

  const form = document.getElementById("review-form");
  if (form) {
    const error = document.getElementById("review-error");
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const values = Object.fromEntries(new FormData(form));
      if (!values.final_route) delete values.final_route;
      const button = form.querySelector("button[type=submit]");
      button.disabled = true;
      const response = await fetch(`/api/claims/${data.id}/review`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(values),
      });
      if (response.ok) { location.reload(); return; }
      let body = {};
      try { body = await response.json(); } catch { /* not JSON */ }
      error.textContent = body.detail || "The decision was not recorded.";
      error.hidden = false;
      button.disabled = false;
    });
  }
})();
