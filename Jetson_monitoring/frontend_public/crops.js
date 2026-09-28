/*
  crops.js -- fills in the RECENT SIGHTINGS feed.

  Runs updateCrops() every 2 seconds, forever (see the very bottom of this
  file). Each time, it:
    1. Asks the Python backend (server.py) for the most recent sightings, via
       fetch('/recent_crops'). The backend answers with a JSON array -- see
       server.py's PUBLIC_CROP_FIELDS for the exact, deliberately short list
       of fields it ever sends (species, confidence, size, time, image URL --
       nothing else).
    2. Turns each entry into a small thumbnail + caption (see renderCrop()).
    3. Replaces #cropsFeed's contents with the new list.

  Same overall shape as frontend_private/status.js, just simpler -- one
  endpoint, one list, no warning colors.
*/

const cropsFeed = document.getElementById("cropsFeed");
const cropsEmpty = document.getElementById("cropsEmpty");

// ===== FORMATTING HELPERS =====

// "small_ctenophore" -> "Small ctenophore" -- the raw class_label is a
// classifier-internal identifier (underscores, all-lowercase); this is the
// one place that gets turned into something readable for a public visitor.
function fmtSpecies(label) {
  if (!label) return "Unidentified";
  const text = label.replace(/_/g, " ");
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function fmtConfidence(confidence) {
  if (confidence === null || confidence === undefined) return "";
  return Math.round(confidence * 100) + "%";
}

function fmtSize(mm2) {
  if (mm2 === null || mm2 === undefined) return "";
  return mm2.toFixed(1) + " mm²";
}

// Same "seconds ago" logic as frontend_private/status.js's fmtAge -- kept
// here rather than shared since these two frontends are deliberately
// independent (see this project's README for why).
function fmtAge(isoTimestamp) {
  if (!isoTimestamp) return "";
  const seconds = (Date.now() - new Date(isoTimestamp).getTime()) / 1000;
  if (seconds < 0) return "just now"; // clock skew between machines, not worth explaining to a visitor
  if (seconds < 120) return Math.round(seconds) + "s ago";
  if (seconds < 7200) return (seconds / 60).toFixed(1) + "m ago";
  return (seconds / 3600).toFixed(1) + "h ago";
}

// ===== RENDERING =====

function renderCrop(crop) {
  const caption = [fmtSpecies(crop.species), fmtConfidence(crop.confidence)]
    .filter(Boolean)
    .join(" · ");
  const details = [fmtSize(crop.size_mm2), fmtAge(crop.time)]
    .filter(Boolean)
    .join(" · ");
  return `
    <div class="crop-entry">
      <img class="crop-thumb" src="${crop.image_url}" alt="${caption}">
      <div class="crop-caption">
        <span class="crop-species">${caption}</span>
        <span class="crop-details">${details}</span>
      </div>
    </div>
  `;
}

function updateCrops() {
  fetch("/recent_crops")
    .then((response) => response.json())
    .then((crops) => {
      if (crops.length === 0) {
        cropsFeed.style.display = "none";
        cropsEmpty.style.display = "block";
        return;
      }
      cropsFeed.style.display = "block";
      cropsEmpty.style.display = "none";
      cropsFeed.innerHTML = crops.map(renderCrop).join("");
    })
    // A single failed poll (e.g. brief network hiccup) isn't worth showing an
    // error for -- the feed just keeps whatever it last had until the next
    // poll succeeds.
    .catch(() => {});
}

updateCrops();
setInterval(updateCrops, 2000);
