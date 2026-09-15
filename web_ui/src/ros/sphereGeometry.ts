import * as THREE from "three";

export function contactTriangleQuality(points: THREE.Vector3[]): number | null {
  if (points.length !== 3) return null;
  const sideSquares = [
    points[0].distanceToSquared(points[1]),
    points[1].distanceToSquared(points[2]),
    points[2].distanceToSquared(points[0]),
  ];
  const denominator = sideSquares.reduce((sum, side) => sum + side, 0);
  if (denominator <= 1e-12) return null;
  const crossLength = new THREE.Vector3()
    .subVectors(points[1], points[0])
    .cross(new THREE.Vector3().subVectors(points[2], points[0]))
    .length();
  return 2 * Math.sqrt(3) * crossLength / denominator;
}

export function averageCenters(centers: THREE.Vector3[]): THREE.Vector3 | null {
  if (centers.length === 0) return null;
  return centers.reduce((sum, center) => sum.add(center), new THREE.Vector3())
    .multiplyScalar(1 / centers.length);
}

export function sphereEstimationFingerIds(graspType: number, activeFingerIds: number[]): number[] {
  return graspType === 3 ? activeFingerIds.slice(0, 3) : [];
}

export function sphereEstimationConfigurations(
  graspType: number,
  activeFingerIds: number[],
): { fingerIds: number[]; disambiguationFinger?: number }[] {
  if (graspType >= 4) {
    return [
      { fingerIds: [1, 2, 3], disambiguationFinger: 4 },
      { fingerIds: [1, 2, 4], disambiguationFinger: 3 },
      { fingerIds: [1, 3, 4], disambiguationFinger: 2 },
    ];
  }
  const fingerIds = sphereEstimationFingerIds(graspType, activeFingerIds);
  return fingerIds.length === 3 ? [{ fingerIds }] : [];
}

export function fitSphereCenterFromContactTriangle(
  points: THREE.Vector3[],
  radius: number,
  reference?: THREE.Vector3,
  onFailure?: (reason: string) => void,
  circumradiusTolerance = 0,
  pointLabels = points.map((_, index) => index + 1),
  disambiguationPoint?: THREE.Vector3,
): THREE.Vector3 | null {
  const fail = (reason: string) => {
    onFailure?.(reason);
    return null;
  };
  if (points.length < 3) return fail(`not enough FK points (${points.length}/3)`);
  if (radius <= 0) return fail(`invalid center distance (${radius})`);

  let triangle: [THREE.Vector3, THREE.Vector3, THREE.Vector3] | null = null;
  let triangleIndices: [number, number, number] | null = null;
  let bestQuality = -1;
  for (let first = 1; first < points.length - 1; first += 1) {
    for (let second = first + 1; second < points.length; second += 1) {
      const quality = contactTriangleQuality([points[0], points[first], points[second]]);
      if (quality !== null && Number.isFinite(quality) && quality > bestQuality) {
        bestQuality = quality;
        triangle = [points[0], points[first], points[second]];
        triangleIndices = [pointLabels[0], pointLabels[first], pointLabels[second]];
      }
    }
  }
  if (!triangle || !triangleIndices) return fail("no valid thumb-based FK triangle");

  const [p1, p2, p3] = triangle;
  const u = new THREE.Vector3().subVectors(p2, p1);
  const v = new THREE.Vector3().subVectors(p3, p1);
  const normal = new THREE.Vector3().crossVectors(u, v);
  const normalSquared = normal.lengthSq();
  if (normalSquared < 1e-12) {
    return fail(`FK points ${triangleIndices.join(",")} are nearly collinear`);
  }

  const circumcenter = p1.clone().add(
    v.clone().cross(normal).multiplyScalar(u.lengthSq())
      .add(normal.clone().cross(u).multiplyScalar(v.lengthSq()))
      .multiplyScalar(1 / (2 * normalSquared)),
  );
  const circumradius = circumcenter.distanceTo(p1);
  const heightSquared = radius * radius - circumradius * circumradius;
  if (circumradius - radius > circumradiusTolerance + 1e-8) {
    return fail(
      `circumradius ${Math.round(circumradius * 1000 * 10) / 10}mm exceeds ${Math.round(radius * 1000 * 10) / 10}mm (FK points ${triangleIndices.join(",")})`,
    );
  }

  const offset = normal.normalize().multiplyScalar(Math.sqrt(Math.max(0, heightSquared)));
  const candidates = [circumcenter.clone().add(offset), circumcenter.clone().sub(offset)];
  if (disambiguationPoint) {
    return Math.abs(candidates[1].distanceTo(disambiguationPoint) - radius)
      < Math.abs(candidates[0].distanceTo(disambiguationPoint) - radius)
      ? candidates[1]
      : candidates[0];
  }
  return reference && candidates[1].distanceToSquared(reference) < candidates[0].distanceToSquared(reference)
    ? candidates[1]
    : candidates[0];
}
