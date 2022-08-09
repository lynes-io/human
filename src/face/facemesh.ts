/**
 * BlazeFace, FaceMesh & Iris model implementation
 *
 * Based on:
 * - [**MediaPipe BlazeFace**](https://drive.google.com/file/d/1f39lSzU5Oq-j_OXgS67KfN5wNsoeAZ4V/view)
 * - Facial Spacial Geometry: [**MediaPipe FaceMesh**](https://drive.google.com/file/d/1VFC_wIpw4O7xBOiTgUldl79d9LA-LsnA/view)
 * - Eye Iris Details: [**MediaPipe Iris**](https://drive.google.com/file/d/1bsWbokp9AklH2ANjCfmjqEzzxO1CNbMu/view)
 */

import { log, now } from '../util/util';
import { loadModel } from '../tfjs/load';
import * as tf from '../../dist/tfjs.esm.js';
import * as blazeface from './blazeface';
import * as util from './facemeshutil';
import * as coords from './facemeshcoords';
import { env } from '../util/env';
import type { GraphModel, Tensor } from '../tfjs/types';
import type { FaceResult, FaceLandmark, Point } from '../result';
import type { Config } from '../config';

type DetectBox = { startPoint: Point, endPoint: Point, landmarks: Array<Point>, confidence: number };

const cache = {
  boxes: [] as DetectBox[],
  skipped: Number.MAX_SAFE_INTEGER,
  timestamp: 0,
};

let model: GraphModel | null = null;

export async function predict(input: Tensor, config: Config): Promise<FaceResult[]> {
  // reset cached boxes
  const skipTime = (config.face.detector?.skipTime || 0) > (now() - cache.timestamp);
  const skipFrame = cache.skipped < (config.face.detector?.skipFrames || 0);
  if (!config.skipAllowed || !skipTime || !skipFrame || cache.boxes.length === 0) {
    cache.boxes = await blazeface.getBoxes(input, config); // get results from blazeface detector
    cache.timestamp = now();
    cache.skipped = 0;
  } else {
    cache.skipped++;
  }
  const faces: Array<FaceResult> = [];
  const newCache: Array<DetectBox> = [];
  let id = 0;
  for (let i = 0; i < cache.boxes.length; i++) {
    const box = cache.boxes[i];
    const face: FaceResult = { // init face result
      id: id++,
      mesh: [],
      meshRaw: [],
      box: [0, 0, 0, 0],
      boxRaw: [0, 0, 0, 0],
      score: 0,
      boxScore: 0,
      faceScore: 0,
      // contoursRaw: [],
      // contours: [],
      annotations: {} as Record<FaceLandmark, Point[]>,
    };

    face.boxScore = Math.round(100 * box.confidence) / 100;
    face.box = util.clampBox(box, input);
    face.boxRaw = util.getRawBox(box, input);
    face.score = face.boxScore;
    if (face.score > (config.face.detector?.minConfidence || 1)) faces.push(face);
    else tf.dispose(face.tensor);
  }
  cache.boxes = newCache; // reset cache
  return faces;
}

export async function load(config: Config): Promise<GraphModel> {
  if (env.initial) model = null;
  if (!model) {
    model = await loadModel(config.face.mesh?.modelPath);
  } else if (config.debug) {
    log('cached model:', model['modelUrl']);
  }
  return model;
}

export const triangulation = coords.TRI468;
export const uvmap = coords.UV468;
