/*
  status.js -- fills in all the text on the dashboard (temperatures, queue sizes,
  connection health, etc).

  START HERE. This file runs updateStatus() once a second, forever (see the very
  bottom of this file). Each time, it:
    1. Asks the Python backend (analyse.py) for the latest numbers, via
       fetch('/status'). The backend answers with JSON -- see
       Monitor/analyse.py's _stream_status() function for the full list of fields
       it can send, or just log `d` (see below) to your browser's console.
    2. Turns those raw numbers into readable text -- see "FORMATTING HELPERS"
       below (fmtAge, fmtSpeed, etc).
    3. Builds a small chunk of HTML for each section, and writes it into the
       matching empty <section> that's already sitting in index.html (see
       "SECTION RENDERERS" below).

  Want to add a new stat to the page? See README.md's "Add a new stat" walkthrough
  -- the short version is: pick the renderer function for the section you want
  (below), add one line using an existing field from the JSON and one of the fmt*
  helpers, and refresh the page.
*/

// ===== FORMATTING HELPERS =====
// Small functions that turn a raw number (or null, if the backend doesn't have a
// reading yet) into text worth putting on screen. None of these talk to the
// network -- they're pure "take a value in, get text out" helpers.

// `v ?? w` below is JS's "nullish coalescing" -- it's `w` only when `v` is exactly
// null or undefined, unlike `v || w` which would also replace a real value of 0.
// Not used everywhere here (some spots need to tell "0" and "no reading" apart on
// purpose), but worth knowing since you'll see `||` used similarly a few times below
// for "value, or 0 if we don't have one yet" (e.g. queue depths before the first
// reading arrives).

function fmt(value, unit) {
  if (value === null || value === undefined) return "N/A";
  return value.toFixed(1) + unit;
}

function overMax(value, max) {
  return value !== null && value !== undefined && max !== null && max !== undefined && value >= max;
}

function nearMax(value, max) {
  return value !== null && value !== undefined && max !== null && max !== undefined && value >= 0.9 * max;
}

// Returns a CSS class name (see style.css's "STATUS COLORS" section) for a value
// relative to its max, or null if it's not near/over -- null means "don't add a
// color class at all," handled by colorize() below.
function statusClass(value, max) {
  if (overMax(value, max)) return "status-danger";
  if (nearMax(value, max)) return "status-warn";
  return null;
}

function flag(value, max) {
  if (overMax(value, max)) return " [DANGER]";
  if (nearMax(value, max)) return " [WARN]";
  return "";
}

// Wraps `text` in a colored <span> if `cls` is a real class name, otherwise
// returns the text plain. Shared by vsMax() below and renderHeader()'s mode line.
function colorize(text, cls) {
  return cls ? `<span class="${cls}">${text}</span>` : text;
}

// "value vs. its max", e.g. a temperature vs. its hardware limit -- the common
// pattern used throughout the DEVICE section below.
function vsMax(value, max, unit) {
  const text = `${fmt(value, unit)} / max ${fmt(max, unit)}${flag(value, max)}`;
  return colorize(text, statusClass(value, max));
}

function fmtSpeed(bytesPerSec) {
  if (bytesPerSec === null || bytesPerSec === undefined) return "N/A";
  if (bytesPerSec >= 1e6) return (bytesPerSec / 1e6).toFixed(2) + " MB/s";
  if (bytesPerSec >= 1e3) return (bytesPerSec / 1e3).toFixed(1) + " KB/s";
  return bytesPerSec.toFixed(0) + " B/s";
}

// Turns "seconds ago" into whichever unit reads best -- nobody wants to read
// "5400s ago" when "1.5h ago" says the same thing more clearly.
function fmtAge(seconds) {
  if (seconds === null || seconds === undefined) return "N/A";
  if (seconds < 120) return seconds.toFixed(0) + "s ago";
  if (seconds < 7200) return (seconds / 60).toFixed(1) + "m ago";
  if (seconds < 172800) return (seconds / 3600).toFixed(1) + "h ago";
  return (seconds / 86400).toFixed(1) + "d ago";
}

function fmtTime(unixSeconds) {
  if (unixSeconds === null || unixSeconds === undefined) return "N/A";
  return new Date(unixSeconds * 1000).toLocaleTimeString();
}

// The backend sends a timestamp (e.g. last_recorded_unix); this turns it into
// "how many seconds ago was that", for fmtAge() above to format.
function age(unixSeconds) {
  if (unixSeconds === null || unixSeconds === undefined) return null;
  return Date.now() / 1000 - unixSeconds;
}

function fmtEta(hours) {
  if (hours === null || hours === undefined) return "N/A";
  if (hours < 48) return hours.toFixed(1) + "h";
  return (hours / 24).toFixed(1) + "d";
}

// Every section (except the untitled header) starts with a bold title line, then
// its stat lines joined with "\n" -- style.css's white-space:pre-wrap on
// #statusBox is what turns those "\n"s into real line breaks on screen.
function sectionHtml(title, lines) {
  return `<span class="section-title">${title}</span>\n` + lines.join("\n");
}

// ===== SECTION RENDERERS =====
// One function per <section> in index.html, each named to match. Every renderer
// takes the parsed JSON from /status (always called `d` for "data") and returns
// an HTML string -- nothing here touches the page directly, that happens once,
// in updateStatus() below.

function renderHeader(d) {
  const modeText = d.system_mode === "live" ? "LIVE" : "BACKLOG";
  const mode = colorize(modeText, d.system_mode === "live" ? null : "status-warn");
  return (
    `<span class="section-title">${d.time}</span>\n` +
    `Mode: ${mode} (${(d.total_backlog || 0).toLocaleString()} pending)\n` +
    `Last recorded: ${fmtTime(d.last_recorded_unix)} (${fmtAge(age(d.last_recorded_unix))})\n` +
    `Last analysed: ${fmtTime(d.last_analysed_unix)} (${fmtAge(age(d.last_analysed_unix))})\n` +
    `Crops in last image: ${d.last_analysed_crops}  Crops in last 24h: ${d.crops_last_24h}\n` +
    `Sampling: every ${d.frame_skip} frame(s)`
  );
}

// Only ever shown while system_mode isn't 'live' -- see updateStatus() below,
// which hides this whole section once there's nothing left to catch up on.
function renderBacklog(d) {
  const fullyLiveIn =
    d.camera_live_catching_up === true
      ? fmtEta(d.camera_live_eta_hours)
      : d.camera_live_catching_up === false
      ? `NOT catching up (record ${fmt(d.record_rate_per_hour, "/hr")} vs analyse ${fmt(d.analyse_rate_per_hour, "/hr")})`
      : "estimating...";

  // .map() below builds one line of text per backlog stage -- "for each stage in
  // the list, turn it into a line of text" is exactly what .map() means; the
  // result is a new array of strings, same length as backlog_stages.
  const stageLines = (d.backlog_stages || []).map((stage) => {
    if (stage.status === "done") return `${stage.label}: done`;
    if (stage.status === "pending") return `${stage.label}: pending`;
    let text = `${stage.label}: active, ${stage.remaining.toLocaleString()} remaining`;
    if (stage.rate_per_hour) {
      text += ` -- ${stage.rate_per_hour.toFixed(0)}/hr, ETA ${fmtEta(stage.eta_hours)}`;
    }
    return text;
  });

  // `...stageLines` "spreads" that array's items in as individual entries of this
  // outer array, instead of nesting one array inside another.
  return sectionHtml("BACKLOG", [`Fully live in: ${fullyLiveIn}`, ...stageLines]);
}

function renderQueue(d) {
  return sectionHtml("QUEUE", [
    `Avg processing time: ${fmt(d.avg_processing_time_s, " s")}`,
    `Crops queue: ${(d.que_crops_depth || 0).toLocaleString()}  ` +
      `Full frames queue: ${(d.que_fullframes_depth || 0).toLocaleString()}`,
  ]);
}

function renderLeak(d) {
  const leakText = d.leak_detected === true ? "!!! DETECTED !!!" : d.leak_detected === false ? "OK" : "N/A";
  return sectionHtml("LEAK DETECTION", [
    `Leak: ${leakText}`,
    `Enclosure humidity: ${fmt(d.bme280_humidity_pct, "%")}  Dew point: ${fmt(d.dew_point_c, " C")}`,
  ]);
}

function renderEnvironmental(d) {
  return sectionHtml("ENVIRONMENTAL", [
    `Bar3XT pressure: ${fmt(d.bar3xt_pressure_mbar, " mbar")}  depth: ${fmt(d.bar3xt_depth_m, " m")}`,
    `Bar3XT temp: ${fmt(d.bar3xt_temp_c, " C")}  DS18B20 temp: ${fmt(d.ds18b20_temp_c, " C")}`,
  ]);
}

function renderDevice(d) {
  return sectionHtml("DEVICE", [
    `Enclosure pressure: ${fmt(d.bme280_pressure_mbar, " mbar")}`,
    `Enclosure temp (BME280): ${fmt(d.bme280_temp_c, " C")}`,
    `Camera: ${vsMax(d.camera_temp_c, d.camera_max_temp_c, " C")}`,
    `Strobe converter: ${vsMax(d.strobe_converter_temp_c, d.strobe_max_temp_c, " C")}`,
    `Strobe driver: ${vsMax(d.strobe_driver_temp_c, d.strobe_max_temp_c, " C")}`,
    `Jetson: ${vsMax(d.jetson_temp_c_mean, d.jetson_max_temp_c, " C")}`,
    `CPU: ${fmt(d.cpu_percent, "%")}  GPU: ${fmt(d.gpu_percent_mean, "%")} (max ${fmt(d.gpu_percent_max, "%")})`,
    `Disk free: device: ${fmt(d.disk_root, " GB")}  ` +
      `ssd1: ${fmt(d.disk_ssd1, " GB")}  ssd2: ${fmt(d.disk_ssd2, " GB")}`,
  ]);
}

function renderConnection(d) {
  const statusText = !d.send_alive
    ? "PROCESS DOWN"
    : d.consecutive_upload_failures > 0
    ? `FAILING (${d.consecutive_upload_failures}x)`
    : "OK";
  return sectionHtml("CONNECTION", [
    `Status: ${statusText}`,
    `Last successful upload: ${fmtAge(d.last_success_age_s)}  Last upload speed: ${fmtSpeed(d.upload_speed_bps)}`,
    `Link speed: ${fmtSpeed(d.link_speed_bps)} (probed ${fmtAge(d.link_speed_probed_age_s)})`,
  ]);
}

// ===== THE POLLING LOOP =====

// `async`/`await` lets this function pause at each `await` until that operation
// finishes, then carry on -- written top-to-bottom like ordinary step-by-step
// code, even though "wait for the network" is involved.
async function updateStatus() {
  try {
    const response = await fetch("/status");
    const d = await response.json();

    document.getElementById("header").innerHTML = renderHeader(d);

    // BACKLOG only makes sense to show while there's actually one to report --
    // once system_mode is 'live', hide the whole section instead of leaving a
    // stale/meaningless "done" state on screen.
    const backlogSection = document.getElementById("backlog");
    if (d.system_mode === "live") {
      backlogSection.style.display = "none";
    } else {
      backlogSection.style.display = "";
      backlogSection.innerHTML = renderBacklog(d);
    }

    document.getElementById("queue").innerHTML = renderQueue(d);
    document.getElementById("leak").innerHTML = renderLeak(d);
    document.getElementById("environmental").innerHTML = renderEnvironmental(d);
    document.getElementById("device").innerHTML = renderDevice(d);
    document.getElementById("connection").innerHTML = renderConnection(d);

    // `classList.toggle(name, condition)` adds the class when `condition` is
    // true and removes it otherwise -- see style.css for what each class does.
    document.getElementById("statusBox").classList.toggle("leak", d.leak_detected === true);
    document.getElementById("connection").classList.toggle("down", !d.send_alive);

    // The video frame and crops strip only get produced by the backend in live
    // mode (see analyse.py) -- hide them in backlog mode rather than leaving a
    // permanently-broken image on screen.
    const isLive = d.system_mode === "live";
    document.getElementById("videoFrame").style.display = isLive ? "block" : "none";
    document.getElementById("livestreamBadge").style.display = isLive ? "block" : "none";
    document.getElementById("cropsPanel").style.display = isLive ? "block" : "none";
  } catch (err) {
    // fetch() or JSON parsing can fail momentarily (e.g. the backend restarting)
    // -- just skip this tick silently and try again in a second, rather than
    // leaving a half-broken page on screen.
  }
}

// Run once immediately, then keep polling forever.
updateStatus();
setInterval(updateStatus, 1000);
