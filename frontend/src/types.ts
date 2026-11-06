// Shapes from docs/api-contract.md
export type Status = "identified" | "ambiguous" | "unknown" | "no_vehicle";
export type Powertrain = "hybrid" | "ev" | "combustion";

export interface Candidate {
  label: string;
  make: string;
  model: string;
  year: number;
  prob: number;
  powertrain: Powertrain;
}

export interface Safety {
  hv_probability: number | null;
  hv_warning: boolean;
  level: "danger" | "caution" | "info";
  message: string;
  actions: string[];
  guide: { title: string; url: string } | null;
  location_vehicle?: string | null;
  battery?: string | null;
  hv_cables?: string | null;
}

export interface Quality {
  brightness: number;
  contrast: number;
  sharpness: number;
  resolution_px: number;
  cut_off: boolean;
  issues: string[];
}

export interface Step {
  key: string;
  title: string;
  caption: string;
  image: string;
}

export interface Pick {
  label: string;
  prob: number;
}

export interface Result {
  request_id: string;
  status: Status;
  headline: string;
  image: { width: number; height: number };
  detection: { box: [number, number, number, number]; score: number } | null;
  other_boxes: number[][];
  prediction: { label: string; make: string; model: string; year: number; body: string; confidence: number } | null;
  hierarchy: { level: "make" | "family" | "vehicle"; name: string; prob: number }[];
  resolved_level: "vehicle" | "family" | "make" | "none";
  top5: Candidate[];
  safety: Safety | null;
  unknown: { score: number; threshold: number; is_unknown: boolean } | null;
  quality: Quality | null;
  reliability: { score: number; reasons: string[] } | null;
  steps: Step[];
  features: {
    dominant_colours: { hex: string; name: string; share: number }[];
    aspect_ratio: number;
    wheel_count: number;
    wheelbase_ratio: number | null;
  } | null;
  comparison: { classical: Pick | null; deep: Pick; hybrid: Pick & { status: Status } } | null;
  timing_ms: { detect: number; classify: number; classical: number; total: number };
  disclaimer: string;
  mock?: boolean;
  brand_evidence?: BrandEvidence | null; // absent in offline mode (the close-up expert runs on the server only)
  views?: ViewSummary[]; // one per photo sent; absent in offline mode
  offline?: boolean;
}

/** What one photo said on its own, before the photos were combined. */
export interface ViewSummary {
  index: number;
  vehicle_found: boolean;
  top: string | null;
  prob: number | null;
  unknown_score: number | null;
  quality_issues: string[];
}

export interface MakeVote {
  make: string;
  prob: number;
}

/** Two experts on the brand: the whole-car classifier (summed per make) and the close-up expert (badge, grille, lights). */
export interface BrandEvidence {
  whole_car: MakeVote[];
  details: MakeVote[];
  combined: MakeVote[];
  weight: number;
  conflict: boolean;
}

export interface Metrics {
  classification?: { top1: number; top5: number; top1_no_crop: number; top1_tta: number };
  detection?: { mean_iou: number; map50: number; map50_95: number };
  efficiency?: { fp32_mb: number; int8_mb: number; fp32_ms: number; int8_ms: number; int8_top1: number; fp32_top1?: number };
  robustness?: { condition: string; severity: number; top1: number }[];
  calibration?: { ece_before: number; ece_after: number; temperature: number };
  safety?: {
    hv_threshold?: number;
    hv_miss_rate: number | null;
    hv_false_alarm_rate: number | null;
    argmax_hv_miss_rate?: number | null;
    known_flagged_unknown_rate?: number | null;
    unknown_catch_rate: number | null;
    by_threshold?: { threshold: number; hv_miss_rate: number; hv_false_alarm_rate: number }[];
  };
  approaches?: { name: string; top1: number | null; top5: number | null }[];
  localisation?: { deep_mean_iou: number; classical_mean_iou: number | null };
  risk_coverage?: { coverage: number; accuracy: number }[];
  top_confusions?: { true: string; predicted: string; count: number }[];
}
