import * as THREE from "three";

export function fitSphereCenterFromContactTriangle(
  points: THREE.Vector3[],
  radius: number,
  reference?: THREE.Vector3,
): THREE.Vector3 | null {
  if (points.length < 3 || radius <= 0) return null;

  let triangle: [THREE.Vector3, THREE.Vector3, THREE.Vector3] | null = null;
  let bestQuality = -1;
  for (let first = 1; first < points.length - 1; first += 1) {
    for (let second = first + 1; second < points.length; second += 1) {
      const sides = [
        points[0].distanceToSquared(points[first]),
        points[first].distanceToSquared(points[second]),
        points[second].distanceToSquared(points[0]),
      ];
      const crossLength = new THREE.Vector3()
        .subVectors(points[first], points[0])
        .cross(new THREE.Vector3().subVectors(points[second], points[0]))
        .length();
      const quality = 2 * Math.sqrt(3) * crossLength / sides.reduce((sum, side) => sum + side, 0);
      if (Number.isFinite(quality) && quality > bestQuality) {
        bestQuality = quality;
        triangle = [points[0], points[first], points[second]];
      }
    }
  }
  if (!triangle) return null;

  const [p1, p2, p3] = triangle;
  const u = new THREE.Vector3().subVectors(p2, p1);
  const v = new THREE.Vector3().subVectors(p3, p1);
  const normal = new THREE.Vector3().crossVectors(u, v);
  const normalSquared = normal.lengthSq();
  if (normalSquared < 1e-12) return null;

  const circumcenter = p1.clone().add(
    v.clone().cross(normal).multiplyScalar(u.lengthSq())
      .add(normal.clone().cross(u).multiplyScalar(v.lengthSq()))
      .multiplyScalar(1 / (2 * normalSquared)),
  );
  const heightSquared = radius * radius - circumcenter.distanceToSquared(p1);
  if (heightSquared < -1e-8) return null;

  const offset = normal.normalize().multiplyScalar(Math.sqrt(Math.max(0, heightSquared)));
  const candidates = [circumcenter.clone().add(offset), circumcenter.clone().sub(offset)];
  return reference && candidates[1].distanceToSquared(reference) < candidates[0].distanceToSquared(reference)
    ? candidates[1]
    : candidates[0];
}
