import assert from "node:assert/strict";
import * as THREE from "three";

import { fitSphereCenterFromContactTriangle } from "./sphereGeometry.ts";

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
