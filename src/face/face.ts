/**
 * Face algorithm implementation
 * Uses FaceMesh, Emotion and FaceRes models to create a unified pipeline
 */

import { log, now } from '../util/util';
import { env } from '../util/env';
import * as tf from '../../dist/tfjs.esm.js';
import * as facemesh from './facemesh';
import type { FaceResult } from '../result';
import type { Tensor } from '../tfjs/types';
import type { Human } from '../human';

export const detectFace = async (instance: Human /* instance of human */, input: Tensor): Promise<FaceResult[]> => {
  // run facemesh, includes blazeface and iris
  // eslint-disable-next-line no-async-promise-executor
  const timeStamp: number = now();

  const faceRes: Array<FaceResult> = [];
  instance.state = 'run:face';

  const faces = await facemesh.predict(input, instance.config);
  instance.performance.face = env.perfadd ? (instance.performance.face || 0) + Math.trunc(now() - timeStamp) : Math.trunc(now() - timeStamp);
  if (!input.shape || input.shape.length !== 4) return [];
  if (!faces) return [];
  // for (const face of faces) {
  for (let i = 0; i < faces.length; i++) {
    // instance.analyze('Get Face');

    // is something went wrong, skip the face
    // @ts-ignore possibly undefied
    if (!faces[i].tensor || faces[i].tensor['isDisposedInternal']) {
      log('Face object is disposed:', faces[i].tensor);
      continue;
    }

    // calculate face angles
    // const rotation = faces[i].mesh && (faces[i].mesh.length > 200) ? calculateFaceAngle(faces[i], [input.shape[2], input.shape[1]]) : null;

    // instance.analyze('Finish Face:');

    // optionally return tensor
    // const tensor = instance.config.face.detector?.return ? tf.squeeze(faces[i].tensor) : null;
    // dispose original face tensor
    tf.dispose(faces[i].tensor);
    // delete temp face image
    if (faces[i].tensor) delete faces[i].tensor;
    // combine results
    const res: FaceResult = {
      ...faces[i],
      id: i,
    };
    // if (rotation) res.rotation = rotation;
    // if (tensor) res.tensor = tensor;
    faceRes.push(res);
    // instance.analyze('End Face');
  }
  // instance.analyze('End FaceMesh:');
  if (instance.config.async) {
    if (instance.performance.face) delete instance.performance.face;
    if (instance.performance.age) delete instance.performance.age;
    if (instance.performance.gender) delete instance.performance.gender;
    if (instance.performance.emotion) delete instance.performance.emotion;
  }
  return faceRes;
};
