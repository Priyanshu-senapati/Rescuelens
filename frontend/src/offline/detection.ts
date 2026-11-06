// Same rule as vehicle_scores in backend/app/models.py, so a fine-tuned detector works offline too.
const VEHICLE_CLASSES = [2, 5, 7]; // COCO car, bus, truck

/** Which output rows hold vehicle scores: car/bus/truck on the 80-class COCO head, every class on our fine-tuned head. */
export function vehicleRows(rows: number): number[] {
  const classes = rows - 4;
  return classes === 80 ? VEHICLE_CLASSES.map((c) => 4 + c) : Array.from({ length: classes }, (_, i) => 4 + i);
}

/** Best vehicle score for one anchor. `data` is rows x count, flattened by row. */
export function vehicleScore(data: ArrayLike<number>, count: number, anchor: number, rows: number[]): number {
  let score = 0;
  for (const row of rows) score = Math.max(score, data[row * count + anchor]);
  return score;
}
