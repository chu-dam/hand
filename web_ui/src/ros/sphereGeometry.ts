import * as THREE from "three";

export interface ContactPolygonEstimate {
  center: THREE.Vector3;
  contactPoints: THREE.Vector3[];
  axisU: THREE.Vector3;
  axisV: THREE.Vector3;
  normal: THREE.Vector3;
  area: number;
  equivalentRadius: number;
}

function bestFitPlaneBasis(
  points: THREE.Vector3[],
  origin: THREE.Vector3,
): [THREE.Vector3, THREE.Vector3, THREE.Vector3] | null {
  const covariance = Array.from({ length: 3 }, () => Array(3).fill(0) as number[]);
  points.forEach((point) => {
    const value = point.clone().sub(origin).toArray();
    for (let row = 0; row < 3; row += 1) {
      for (let column = 0; column < 3; column += 1) {
        covariance[row][column] += value[row] * value[column];
      }
    }
  });
  const eigenvectors = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
  for (let iteration = 0; iteration < 12; iteration += 1) {
    const pairs = [[0, 1], [0, 2], [1, 2]];
    const [p, q] = pairs.reduce((best, pair) => (
      Math.abs(covariance[pair[0]][pair[1]])
        > Math.abs(covariance[best[0]][best[1]]) ? pair : best
    ));
    if (Math.abs(covariance[p][q]) <= 1e-15) break;
    const angle = 0.5 * Math.atan2(
      2 * covariance[p][q],
      covariance[q][q] - covariance[p][p],
    );
    const cosine = Math.cos(angle);
    const sine = Math.sin(angle);
    const app = covariance[p][p];
    const aqq = covariance[q][q];
    const apq = covariance[p][q];
    for (let index = 0; index < 3; index += 1) {
      if (index === p || index === q) continue;
      const aip = covariance[index][p];
      const aiq = covariance[index][q];
      covariance[index][p] = covariance[p][index] = cosine * aip - sine * aiq;
      covariance[index][q] = covariance[q][index] = sine * aip + cosine * aiq;
    }
    covariance[p][p] = cosine ** 2 * app - 2 * sine * cosine * apq + sine ** 2 * aqq;
    covariance[q][q] = sine ** 2 * app + 2 * sine * cosine * apq + cosine ** 2 * aqq;
    covariance[p][q] = covariance[q][p] = 0;
    for (let row = 0; row < 3; row += 1) {
      const vip = eigenvectors[row][p];
      const viq = eigenvectors[row][q];
      eigenvectors[row][p] = cosine * vip - sine * viq;
      eigenvectors[row][q] = sine * vip + cosine * viq;
    }
  }
  const order = [0, 1, 2].sort(
    (left, right) => covariance[right][right] - covariance[left][left],
  );
  if (covariance[order[1]][order[1]] <= 1e-12) return null;
  const vector = (column: number) => new THREE.Vector3(
    eigenvectors[0][column],
    eigenvectors[1][column],
    eigenvectors[2][column],
  ).normalize();
  const axisU = vector(order[0]);
  const axisV = vector(order[1]);
  const normal = axisU.clone().cross(axisV).normalize();
  return [axisU, axisV, normal];
}

export function estimateContactPolygon3D(
  points: THREE.Vector3[],
  fingertipRadius = 0.009,
): ContactPolygonEstimate | null {
  if (points.length < 3) return null;
  const planeOrigin = points.reduce(
    (sum, point) => sum.add(point),
    new THREE.Vector3(),
  ).multiplyScalar(1 / points.length);
  if (!Number.isFinite(fingertipRadius) || fingertipRadius <= 0) return null;
  const basis = bestFitPlaneBasis(points, planeOrigin);
  if (!basis) return null;
  const [axisU, axisV, normal] = basis;
  const contacts = points.map((point) => {
    const offset = point.clone().sub(planeOrigin);
    const u = offset.dot(axisU);
    const v = offset.dot(axisV);
    const normalDistance = offset.dot(normal);
    const sectionRadiusSquared = fingertipRadius ** 2 - normalDistance ** 2;
    const distance = Math.hypot(u, v);
    if (sectionRadiusSquared <= 0) return null;
    const sectionRadius = Math.sqrt(sectionRadiusSquared);
    if (distance <= sectionRadius) return null;
    const contactU = u * (1 - sectionRadius / distance);
    const contactV = v * (1 - sectionRadius / distance);
    return {
      point: planeOrigin.clone()
        .addScaledVector(axisU, contactU)
        .addScaledVector(axisV, contactV),
      u: contactU,
      v: contactV,
    };
  });
  if (contacts.some((contact) => contact === null)) return null;
  const ordered = (contacts as { point: THREE.Vector3; u: number; v: number }[]).sort(
    (left, right) => Math.atan2(left.v, left.u) - Math.atan2(right.v, right.u),
  );
  let twiceArea = 0;
  let centerU = 0;
  let centerV = 0;
  for (let index = 0; index < ordered.length; index += 1) {
    const current = ordered[index];
    const next = ordered[(index + 1) % ordered.length];
    const cross = current.u * next.v - next.u * current.v;
    twiceArea += cross;
    centerU += (current.u + next.u) * cross;
    centerV += (current.v + next.v) * cross;
  }
  if (Math.abs(twiceArea) <= 1e-12) return null;
  const area = Math.abs(twiceArea) / 2;
  const center = planeOrigin.clone()
    .addScaledVector(axisU, centerU / (3 * twiceArea))
    .addScaledVector(axisV, centerV / (3 * twiceArea));
  return {
    center,
    contactPoints: ordered.map((contact) => contact.point),
    axisU,
    axisV,
    normal,
    area,
    equivalentRadius: Math.sqrt(area / Math.PI),
  };
}

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

export function sphereEstimationConfigurations(
  graspType: number,
  activeFingerIds: number[],
): { fingerIds: number[]; disambiguationFinger?: number }[] {
  const active = [...new Set(activeFingerIds)]
    .filter((finger) => finger >= 1 && finger <= 4);
  if (graspType < 3 || !active.includes(1) || active.length < 3) return [];

  const others = active.filter((finger) => finger !== 1);
  const configurations: { fingerIds: number[]; disambiguationFinger?: number }[] = [];
  for (let first = 0; first < others.length - 1; first += 1) {
    for (let second = first + 1; second < others.length; second += 1) {
      const pair = [others[first], others[second]];
      configurations.push({
        fingerIds: [1, ...pair],
        disambiguationFinger: others.find((finger) => !pair.includes(finger)),
      });
    }
  }
  return configurations;
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
