/*
  media.js -- keeps the video frame and crops-strip <img> elements up to date.

  This is the simpler of the two JS files here (no fetch(), no JSON, no async) --
  a good place to start if you're new to this codebase. It just changes an <img>'s
  `src` attribute on a timer; the browser does the rest (downloads the new image,
  swaps it in).

  status.js (the other file) does something different: it asks the backend for
  numbers/text and builds HTML out of them. This file only ever deals with images.
*/

// Grabbing these once, at the top, instead of calling document.getElementById()
// every single time we need them below -- small optimization, and it means each
// function below reads a bit shorter.
const videoFrame = document.getElementById("videoFrame");
const cropsStripImg = document.getElementById("cropsStrip");
const cropsPlaceholder = document.getElementById("cropsPlaceholder");

// Fetches the current video frame from the backend. Note this is NOT a video
// stream in the usual sense -- there's no persistent connection. Every 400ms we
// just ask for "whatever the latest frame is right now" and swap it in. That's
// deliberate: a real streaming connection can only ever fall further behind if
// the network or browser is ever briefly slow, since it has to show every frame
// it already received, in order, before it can show a newer one. Asking fresh
// each time always gets the current frame, nothing piles up.
//
// The `?t=...` on the end isn't a real parameter the backend looks at -- it's
// there purely so the browser sees a "different" URL each time and actually
// re-downloads the image, instead of silently reusing a cached copy of the last
// one it fetched.
function updateFrame() {
  videoFrame.src = "/latest_frame.jpg?t=" + Date.now();
}

// Same cache-busting trick as updateFrame() above, just for the crops strip.
// Polled slower (once a second, not 400ms) because labeled crops show up far less
// often than raw camera frames -- there's nothing to gain from checking faster.
function updateCropsStrip() {
  cropsStripImg.src = "/latest_crops_strip.jpg?t=" + Date.now();
}

// The backend's crops-strip endpoint returns a 404 (no image body) until at least
// one labeled crop exists. `onload` fires when a real image successfully loads;
// `onerror` fires on that 404. We use those two events to swap between showing
// the actual strip image and a plain "waiting..." message, so the page never
// shows a broken-image icon.
cropsStripImg.onload = function () {
  cropsStripImg.style.display = "block";
  cropsPlaceholder.style.display = "none";
};
cropsStripImg.onerror = function () {
  cropsStripImg.style.display = "none";
  cropsPlaceholder.style.display = "block";
};

// Run once immediately (so the page has content right away, not just a blank
// spot for the first 400ms/1000ms), then keep running on a timer forever.
updateFrame();
setInterval(updateFrame, 400);

updateCropsStrip();
setInterval(updateCropsStrip, 1000);
