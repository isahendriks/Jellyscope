/*
  media.js -- keeps the live view (raw camera frame + its green sighting boxes)
  up to date.

  #liveFrame is a plain, undecorated frame -- no boxes baked into its pixels.
  Boxes are drawn separately, on every poll, into #liveFrameBoxes (an SVG
  positioned in exact registration over the image, see style.css), from
  live_frame.json -- crop coordinates and species labels the Jetson computed
  for this exact frame. Drawing them client-side (instead of receiving an
  already-annotated image, which is how this used to work) means this page
  controls how sightings look -- color, labels, whatever's added later --
  independently of the Jetson's own pipeline.
*/

const liveFrame = document.getElementById("liveFrame");
const liveFrameBoxes = document.getElementById("liveFrameBoxes");

// Fetches the current live frame from the backend (server.py, reading
// jellyscope_incoming/live_frame/live_frame.jpg on server-lab). Note this is
// NOT a video stream in the usual sense -- there's no persistent connection.
// Every 2 seconds we just ask for "whatever the latest frame is right now"
// and swap it in. That's deliberate: a real streaming connection can only
// ever fall further behind if the network or browser is ever briefly slow,
// since it has to show every frame it already received, in order, before it
// can show a newer one. Asking fresh each time always gets the current
// frame, nothing piles up.
//
// 2 seconds (not faster) because the Jetson only overwrites live_frame.jpg
// roughly every 3 seconds -- polling much faster would just re-download the
// same bytes.
//
// The `?t=...` on the end isn't a real parameter the backend looks at -- it's
// there purely so the browser sees a "different" URL each time and actually
// re-downloads the image, instead of silently reusing a cached copy of the
// last one it fetched.
// Loads the next frame into an off-DOM Image first, and only swaps it into
// #liveFrame once it's actually finished loading. Setting #liveFrame's own
// src directly to a URL that then 404s (the Jetson occasionally catches
// live_frame.jpg mid-overwrite -- see server.py's live_frame() comment) is
// what makes a browser replace it with a broken-image icon; probing first
// means a failed poll just leaves the last good frame on screen instead,
// until a later poll succeeds.
function updateLiveFrame() {
  const probe = new Image();
  const url = "/live_frame.jpg?t=" + Date.now();
  probe.onload = () => {
    // Reuses the exact same URL probe already fetched -- server.py allows a
    // short-lived cache for this response specifically so this is an instant
    // local reuse, not a second real network request.
    liveFrame.src = url;
  };
  // onerror deliberately does nothing -- see comment above.
  probe.src = url;
}

// "small_ctenophore" -> "Small ctenophore". Same rule as crops.js's
// fmtSpecies -- duplicated rather than imported/shared: these two files are
// meant to be readable independently (see index.html's load-order comment),
// and it's three lines.
function fmtSpecies(label) {
  if (!label) return "Unidentified";
  const text = label.replace(/_/g, " ");
  return text.charAt(0).toUpperCase() + text.slice(1);
}

// A label's on-screen size shouldn't depend on how large #liveFrameBoxes
// happens to be rendered (a phone vs. a wide monitor) -- but SVG font-size is
// in viewBox units, which DO get stretched by that rendered size. This
// converts "I want N real screen pixels tall" into the matching viewBox-unit
// value, given how many screen pixels one viewBox unit currently covers.
// (.crop-box's stroke-width doesn't need this same treatment -- CSS's
// vector-effect: non-scaling-stroke handles that one natively.)
const LABEL_SCREEN_PX = 13;

function renderBoxes(data) {
  liveFrameBoxes.setAttribute("viewBox", `0 0 ${data.image_width} ${data.image_height}`);

  const scale = liveFrameBoxes.getBoundingClientRect().width / data.image_width;
  const labelSize = LABEL_SCREEN_PX / scale;

  liveFrameBoxes.innerHTML = data.crops.map((crop) => {
    const x = crop.crop_x0;
    const y = crop.crop_y0;
    const width = crop.crop_x1 - crop.crop_x0;
    const height = crop.crop_y1 - crop.crop_y0;
    const label = crop.class_confidence == null
      ? fmtSpecies(crop.class_label)
      : `${fmtSpecies(crop.class_label)} ${Math.round(crop.class_confidence * 100)}%`;
    return `
      <rect x="${x}" y="${y}" width="${width}" height="${height}" class="crop-box" />
      <text x="${x}" y="${Math.max(labelSize, y - 4)}" class="crop-box-label"
            font-size="${labelSize}" stroke-width="${labelSize * 0.2}">${label}</text>
    `;
  }).join("");
}

function updateLiveFrameBoxes() {
  fetch("/live_frame.json")
    .then((response) => response.json())
    .then(renderBoxes)
    // A missed poll just leaves the last-drawn boxes in place until the next
    // one succeeds -- same "don't error over one hiccup" rule as crops.js.
    .catch(() => {});
}

// One tick updates both together -- the frame and its boxes are meant to
// match, so they're fetched on the same cadence rather than two independent
// timers that could drift apart request-by-request.
function tick() {
  updateLiveFrame();
  updateLiveFrameBoxes();
}

tick();
setInterval(tick, 2000);
