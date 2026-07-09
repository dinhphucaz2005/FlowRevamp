"""
Global configuration for the Flowchart Digitization pipeline.
All paths and tunable parameters are centralised here.
"""

from pathlib import Path

# ── Project root ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

# ── Data directories (auto-created by the orchestrator) ──────
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
STEP1_DIR = DATA_DIR / "step1_preprocessed"
STEP2_DIR = DATA_DIR / "step2_nodes"
STEP3_DIR = DATA_DIR / "step3_edges"
STEP4_DIR = DATA_DIR / "step4_ocr"
OUTPUT_DIR = DATA_DIR / "output"

ALL_DIRS = [INPUT_DIR, STEP1_DIR, STEP2_DIR, STEP3_DIR,
            STEP4_DIR, OUTPUT_DIR]

# ── Step 1: Preprocessing ────────────────────────────────────
ADAPTIVE_THRESH_BLOCK_SIZE = 15   # Must be odd
ADAPTIVE_THRESH_C_HIGH = 4        # High threshold C for clean node boundaries
ADAPTIVE_THRESH_C_LOW = 4          # Low threshold C to preserve faint lines
MORPH_KERNEL_SIZE = (3, 3)        # Kernel for morphological ops

# ── Step 2: Node detection (Iterative Hough + RANSAC) ────────
MIN_NODE_AREA = 500               # Ignore nodes smaller than this (px²)
RANSAC_N_ITER = 200               # RANSAC iterations per shape fit
RANSAC_INLIER_THRESH = 3.0        # Max distance (px) to count as inlier
RANSAC_MAX_ITERATIONS = 20         # Max extract-and-repeat loops
RANSAC_MASK_MARGIN = 5            # Px margin when masking out a found shape
RANSAC_HOUGH_THRESHOLD = 30       # Hough accumulator threshold (step 2)
RANSAC_HOUGH_MIN_LEN = 20         # Min line length for Hough (step 2)
RANSAC_HOUGH_MAX_GAP = 10         # Max gap between line segments (step 2)

# Shape classification thresholds
CIRCULARITY_THRESHOLD = 0.85      # Above this → circle/connector
MAX_NODE_WIDTH = None             # Optional absolute limit for node width (px)
MAX_NODE_HEIGHT = None            # Optional absolute limit for node height (px)
NODE_SIZE_TOLERANCE = 2        # Allow nodes up to 100% larger than the MAX size (2.0x multiplier)
MAX_NODE_SIZE_RATIO = 0.5         # Fallback: max node size as a fraction of image size
MIN_NODE_SOLIDITY = 0.75          # Reject concave background holes (real shapes are convex)

# ── Step 3: Edge detection ───────────────────────────────────
HOUGH_RHO = 1
HOUGH_THETA_DIVISOR = 180         # np.pi / this value
HOUGH_THRESHOLD = 50
HOUGH_MIN_LINE_LENGTH = 30
HOUGH_MAX_LINE_GAP = 15
ARROW_TEMPLATE_SIZE = 20          # px side for arrowhead template
EDGE_NODE_SNAP_DISTANCE = 40      # Max px distance to snap line-end → node

# ── Step 4: OCR ──────────────────────────────────────────────
ENABLE_OCR = False                # Set to False to temporarily disable OCR
OCR_LANGUAGES = ["en", "vi"]      # EasyOCR language list
OCR_ROI_PADDING_RATIO = 0.08      # 8 % padding around each node crop
OCR_GPU = True                    # Use GPU for EasyOCR


# ── Visualisation ────────────────────────────────────────────
VIS_NODE_COLOR = (10, 161, 123)   # Green/teal bounding boxes
VIS_EDGE_COLOR = (255, 0, 255)    # Magenta connection lines
VIS_BAD_NODE_COLOR = (0, 0, 255)  # Red for bad nodes/artifacts
VIS_TEXT_COLOR = (255, 255, 255)  # White text labels
VIS_TEXT_BG_COLOR = (30, 30, 30)  # Dark charcoal background for label box
VIS_LINE_THICKNESS = 5
VIS_FONT_SCALE = 0.55
