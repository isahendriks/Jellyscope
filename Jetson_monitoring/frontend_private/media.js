/*
  media.js -- keeps the right-hand pane's <img> up to date.

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
const liveViewImage = document.getElementById("liveViewImage");
const liveViewContainer = document.getElementById("liveViewContainer");
const liveViewPlaceholder = document.getElementById("liveViewPlaceholder");
const livestreamBadge = document.getElementById("livestreamBadge");
const liveTabs = document.querySelectorAll(".liveTab");
const videoPane = document.getElementById("videoPane");
const liveTabsBar = document.getElementById("liveTabs"); // the bar itself, not
// the individual .liveTab buttons above -- only need its own rendered height

// #liveViewContainer's height was, through three different attempts (a flexbox
// flex-basis trick, then position:absolute, then CSS Grid's auto/1fr row
// tracks), left for CSS to compute -- each tested clean in an automated
// Chromium instance but still failed live for at least one real user (2026-08),
// consistently: only the top portion of whatever was showing stayed visible,
// with no scrollbar to reach the rest, for BOTH a square photo (Preprocessed)
// and a tall multi-row grid (Crops) alike. Two very differently-shaped kinds of
// content failing identically is the tell -- it ruled out "the image's own
// aspect ratio confusing the container" (the working theory behind all three
// earlier attempts) and pointed at something upstream of any of them: the
// container was never actually being bounded to the real visible viewport in
// that browser in the first place (a plausible cause: OS display scaling or a
// 100vh/actual-viewport mismatch, neither of which is reproducible in this
// codebase's own headless test environment).
//
// Measuring and setting an explicit pixel height here sidesteps all of that at
// once: #videoPane.clientHeight is always the real, already-laid-out box
// (unaffected by whatever CSS mechanism produced it, or any DPI/zoom quirk in
// resolving it), and an inline pixel height on the container can't be circular
// or ambiguous the way a CSS percentage or viewport unit can -- there's no
// layout algorithm left to get it wrong. Re-runs on any size change to
// #videoPane (ResizeObserver), not just window resizes -- DevTools
// docking/undocking, browser zoom, and sidebar content height changes all
// change #videoPane's size without necessarily firing a window "resize" event.
function resizeLiveViewContainer() {
  const availableHeight = videoPane.clientHeight - liveTabsBar.offsetHeight - 12; // 12px = #liveTabs' margin-bottom
  if (availableHeight > 0) {
    liveViewContainer.style.height = availableHeight + "px";
  }
}
new ResizeObserver(resizeLiveViewContainer).observe(videoPane);
resizeLiveViewContainer();

// Every view the right-hand pane can show, one at a time, picked by the tab bar --
// the fully-processed/boxed live frame, the three biofouling-background-model debug
// views (background_model.py's EMA subtraction -- raw/background/raw-minus-
// background), and the last crops strip. One lookup table instead of an if/else
// chain scattered across updateLiveView()/the click handler below.
//
// intervalMs differs per view: "livestream" changes once per PREPROCESS step (the
// main camera feed), the rest change once per processed frame (a coarser cadence),
// so polling them at the same 400ms as the livestream would just re-fetch an
// unchanged image most of the time -- see scheduleLiveView() below for how this
// drives a single self-rescheduling timer instead of several fixed setIntervals.
const LIVE_VIEWS = {
  livestream: { url: "/latest_frame.jpg", placeholder: "Waiting for live frame...", intervalMs: 400 },
  raw: { url: "/latest_raw_frame.jpg", placeholder: "Waiting for raw camera frame...", intervalMs: 400 },
  original: { url: "/bg_debug_original.jpg", placeholder: "Waiting for background model...", intervalMs: 1000 },
  background: { url: "/bg_debug_background.jpg", placeholder: "Waiting for background model...", intervalMs: 1000 },
  subtracted: { url: "/bg_debug_subtracted.jpg", placeholder: "Waiting for background model...", intervalMs: 1000 },
  crops: { url: "/latest_crops_strip.jpg", placeholder: "Waiting for labeled crops...", intervalMs: 1000 },
};
let liveView = "livestream";
let liveViewTimer = null;
let liveViewRequestId = 0;

// Cache-busting `?t=...` trick: not a real parameter the backend looks at, it's
// there purely so the browser sees a "different" URL each time and actually
// re-downloads the image, instead of silently reusing a cached copy of the last one
// it fetched. Deliberately fetches only the currently active tab's image, not all
// five on every tick -- the backend serves this over the same field link send.py's
// uploads share (see config.LIVESTREAM_REQUEST_TIMEOUT_S's comment), so there's no
// reason to keep paying for four images nobody's looking at.
function updateLiveView() {
  const requestId = ++liveViewRequestId;
  liveViewPlaceholder.textContent = LIVE_VIEWS[liveView].placeholder;
  liveViewImage.onload = function () {
    if (requestId !== liveViewRequestId) return;
    liveViewImage.style.display = "block";
    liveViewPlaceholder.style.display = "none";
    scheduleNextLiveView();
  };
  liveViewImage.onerror = function () {
    if (requestId !== liveViewRequestId) return;
    liveViewImage.style.display = "none";
    liveViewPlaceholder.style.display = "block";
    scheduleNextLiveView();
  };
  liveViewImage.src = LIVE_VIEWS[liveView].url + "?t=" + Date.now();
}

// Waits for the current image request to finish before polling again. A fixed timer
// could abort a slow JPEG download on a degraded field connection before it loads.
function scheduleNextLiveView() {
  if (liveViewTimer) clearTimeout(liveViewTimer);
  liveViewTimer = setTimeout(scheduleLiveView, LIVE_VIEWS[liveView].intervalMs);
}

function scheduleLiveView() {
  if (liveViewTimer) clearTimeout(liveViewTimer);
  updateLiveView();
}

// The backend's image endpoints return a 404 (no image body) until there's
// something real to show. `onload` fires when a real image successfully loads;
// `onerror` fires on that 404. We use those two events to swap between showing the
// actual image and a plain "waiting..." message, so the page never shows a
// broken-image icon.
// Switches the active tab (highlighted state, which endpoint gets polled, and at
// what rate) and re-fetches immediately, rather than waiting for the next timer
// tick, so the swap feels instant. Exposed as a named function (not just an inline
// click-handler body) so status.js can fall back to "Livestream" by simulating a
// click on that tab -- see its comment -- without duplicating this logic. The
// LIVESTREAM badge only makes sense on the actual livestream tab -- showing it
// while looking at, say, the raw background-model view would misleadingly suggest
// that's the main processed feed too.
//
// Crops gets a different fit mode than the other four: those are single roughly-
// square frames that should always fit entirely within the pane (object-fit:contain,
// the default -- see style.css), but the crops strip is RECENT_CROPS_COUNT tiles
// wrapped into a fixed-width grid (config.RECENT_CROPS_COLUMNS columns), which at
// 100 crops renders far taller than the pane. Shrinking that to fit would make each
// tile illegibly small, so the "scrollable" class switches #liveViewContainer to
// natural-height/scroll instead of contain-and-letterbox -- see style.css's
// #liveViewContainer.scrollable.
function selectLiveView(view) {
  liveTabs.forEach(function (t) { t.classList.toggle("active", t.dataset.view === view); });
  liveView = view;
  livestreamBadge.style.display = view === "livestream" ? "block" : "none";
  liveViewContainer.classList.toggle("scrollable", view === "crops");
  scheduleLiveView();
}

liveTabs.forEach(function (tab) {
  tab.addEventListener("click", function () {
    if (tab.dataset.view !== liveView) selectLiveView(tab.dataset.view);
  });
});

// Starts the polling loop immediately (so the page has content right away, not
// just a blank spot for the first 400ms), then keeps rescheduling itself forever --
// see scheduleLiveView() above.
scheduleLiveView();
