import assert from "node:assert/strict";
import * as THREE from "three";

import {
  averageCenters,
  contactTriangleQuality,
  fitSphereCenterFromContactTriangle,
  sphereEstimationConfigurations,
  sphereEstimationFingerIds,
} from "./sphereGeometry.ts";

assert.deepEqual(averageCenters([
  new THREE.Vector3(1, 2, 3),
  new THREE.Vector3(3, 4, 5),
  new THREE.Vector3(5, 6, 7),
])?.toArray(), [3, 4, 5]);

const equilateral = [0, 2 * Math.PI / 3, 4 * Math.PI / 3].map(
  (angle) => new THREE.Vector3(Math.cos(angle), Math.sin(angle), 0),
);
assert(Math.abs(contactTriangleQuality(equilateral) - 1) < 1e-12);
assert.equal(contactTriangleQuality([
  new THREE.Vector3(0, 0, 0),
  new THREE.Vector3(1, 0, 0),
  new THREE.Vector3(2, 0, 0),
]), 0);

assert.deepEqual(sphereEstimationFingerIds(3, [1, 3, 5]), [1, 3, 5]);
assert.deepEqual(sphereEstimationConfigurations(5, [1, 2, 3, 4, 5]), [
  { fingerIds: [1, 2, 3], disambiguationFinger: 4 },
  { fingerIds: [1, 2, 4], disambiguationFinger: 3 },
  { fingerIds: [1, 3, 4], disambiguationFinger: 2 },
]);

const center = new THREE.Vector3(0.02, -0.01, 0.03);
const radius = 0.0375;
const sectionRadius = 0.03;
const planeOffset = Math.sqrt(radius ** 2 - sectionRadius ** 2);
const points = [
  new THREE.Vector3(sectionRadius, 0, planeOffset),
  new THREE.Vector3(0, sectionRadius, planeOffset),
  new THREE.Vector3(-sectionRadius, 0, planeOffset),
].map((point) => point.add(center));

const estimated = fitSphereCenterFromContactTriangle(points, radius, center);
assert(estimated);
assert(estimated.distanceTo(center) < 1e-12);

const oppositeCenter = center.clone().add(new THREE.Vector3(0, 0, 2 * planeOffset));
const middleTip = center.clone().add(new THREE.Vector3(0, 0, -radius));
const middleDisambiguated = fitSphereCenterFromContactTriangle(
  points,
  radius,
  oppositeCenter,
  undefined,
  0,
  [1, 2, 4],
  middleTip,
);
assert(middleDisambiguated);
assert(middleDisambiguated.distanceTo(center) < 1e-12);

const oversizedSection = [0, 2 * Math.PI / 3, 4 * Math.PI / 3].map(
  (angle) => new THREE.Vector3(0.0476 * Math.cos(angle), 0.0476 * Math.sin(angle), 0),
);
let failure = "";
assert.equal(
  fitSphereCenterFromContactTriangle(
    oversizedSection,
    0.047,
    undefined,
    (reason) => { failure = reason; },
    0,
    [1, 2, 4],
  ),
  null,
);
assert.match(failure, /47\.6mm exceeds 47mm/);
assert.match(failure, /FK points 1,2,4/);
const approximate = fitSphereCenterFromContactTriangle(
  oversizedSection,
  0.047,
  undefined,
  undefined,
  0.002,
);
assert(approximate);
assert(approximate.length() < 1e-12);
