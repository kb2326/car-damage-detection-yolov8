// The customer chat. Messages are inserted with textContent only, never as HTML.
(() => {
  const list = document.getElementById("messages");
  const form = document.getElementById("chat-form");
  const text = document.getElementById("text");
  const photo = document.getElementById("photo");
  const photoLabel = document.getElementById("photo-label");
  const send = document.getElementById("send");
  const start = document.getElementById("start");
  const error = document.getElementById("chat-error");
  const done = document.getElementById("done");
  let session = new URLSearchParams(location.search).get("session");

  function say(who, words) {
    const item = document.createElement("li");
    item.className = who;
    item.textContent = words;
    list.appendChild(item);
    item.scrollIntoView({ block: "end" });
  }

  function show(turn) {
    session = turn.session_id;
    history.replaceState(null, "", `/?session=${encodeURIComponent(session)}`);
    say("agent", turn.message);
    error.hidden = true;
    start.hidden = true;
    if (turn.claim_id) {
      form.hidden = true;
      done.hidden = false;
      done.textContent = `Claim received. Your reference is ${turn.claim_id}. `;
      const link = document.createElement("a");
      link.href = `/claims/${turn.claim_id}`;
      link.textContent = "Open it as the adjuster";
      done.appendChild(link);
      return;
    }
    form.hidden = false;
    const kind = turn.photo_kind ? turn.photo_kind.replaceAll("_", " ") : "";
    photoLabel.hidden = photo.hidden = !turn.photo_kind;
    photoLabel.textContent = kind ? `Add the ${kind} photo` : "";
    (turn.photo_kind ? photo : text).focus();
  }

  async function call(url, options) {
    const response = await fetch(url, options);
    let body = {};
    try { body = await response.json(); } catch { /* not JSON */ }
    if (!response.ok) throw new Error(body.detail || "Something went wrong. Please try again.");
    return body;
  }

  function fail(err) {
    error.textContent = err.message;
    error.hidden = false;
  }

  start.addEventListener("click", () => {
    start.disabled = true;
    call("/api/intake", { method: "POST" }).then(show, fail).finally(() => { start.disabled = false; });
  });

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const file = photo.files.length ? photo.files[0] : null;
    if (!text.value.trim() && !file) return;
    const data = new FormData();
    data.append("text", text.value);
    if (file) data.append("photo", file);
    say("customer", file ? `[photo: ${file.name}] ${text.value}`.trim() : text.value);
    send.disabled = true;
    call(`/api/intake/${encodeURIComponent(session)}/reply`, { method: "POST", body: data })
      .then(show, fail)
      .finally(() => {
        send.disabled = false;
        text.value = "";
        photo.value = "";
      });
  });

  if (session) {
    call(`/api/intake/${encodeURIComponent(session)}`).then(show, () => {
      session = null;
      history.replaceState(null, "", "/");
    });
  }
})();
