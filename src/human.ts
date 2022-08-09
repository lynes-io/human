/**
 * Human main module
 * @default Human Library
 * @summary <https://github.com/vladmandic/human>
 * @author <https://github.com/vladmandic>
 * @copyright <https://github.com/vladmandic>
 * @license MIT
 */

// module imports
import { log, now, mergeDeep, validate } from './util/util';
import { defaults } from './config';
import { env, Env } from './util/env';
import { setModelLoadOptions } from './tfjs/load';
import * as tf from '../dist/tfjs.esm.js';
import * as app from '../package.json';
import * as backend from './tfjs/backend';
import * as face from './face/face';
import * as facemesh from './face/facemesh';
import * as humangl from './tfjs/humangl';
import * as image from './image/image';
import * as interpolate from './util/interpolate';
import * as match from './face/match';
import * as models from './models';
import * as persons from './util/persons';
import * as warmups from './warmup';
// type definitions
import type { Input, Tensor, Config, Result, FaceResult, HandResult, BodyResult, ObjectResult, AnyCanvas, ModelStats } from './exports';
// type exports
export * from './exports';

/** **Human** library main class
 *
 * All methods and properties are available only as members of Human class
 *
 * - Configuration object definition: {@link Config}
 * - Results object definition: {@link Result}
 * - Possible inputs: {@link Input}
 *
 * @param userConfig - {@link Config}
 * @returns instance of {@link Human}
 */
export class Human {
  /** Current version of Human library in *semver* format */
  version: string;

  /** Current configuration
   * - Defaults: [config](https://github.com/vladmandic/human/blob/main/src/config.ts#L262)
   */
  config: Config;

  /** Last known result of detect run
   * - Can be accessed anytime after initial detection
  */
  result: Result;

  /** Current state of Human library
   * - Can be polled to determine operations that are currently executed
   * - Progresses through: 'config', 'check', 'backend', 'load', 'run:<model>', 'idle'
   */
  state: string;

  /** currenty processed image tensor and canvas */
  process: { tensor: Tensor | null, canvas: AnyCanvas | null };

  /** Instance of TensorFlow/JS used by Human
   *  - Can be embedded or externally provided
   * [TFJS API](https://js.tensorflow.org/api/latest/)
   */
  tf;

  /** Object containing environment information used for diagnostics */
  env: Env;

  /** Currently loaded models
   * @internal
   * {@link Models}
  */
  models: models.Models;

  /** Container for events dispatched by Human
   * Possible events:
   * - `create`: triggered when Human object is instantiated
   * - `load`: triggered when models are loaded (explicitly or on-demand)
   * - `image`: triggered when input image is processed
   * - `result`: triggered when detection is complete
   * - `warmup`: triggered when warmup is complete
   * - `error`: triggered on some errors
   */
  events: EventTarget | undefined;
  /** Reference face triangualtion array of 468 points, used for triangle references between points */
  faceTriangulation: number[];
  /** Refernce UV map of 468 values, used for 3D mapping of the face mesh */
  faceUVMap: [number, number][];
  /** Performance object that contains values for all recently performed operations */
  performance: Record<string, number>; // perf members are dynamically defined as needed
  #numTensors: number;
  #analyzeMemoryLeaks: boolean;
  #checkSanity: boolean;
  /** WebGL debug info */
  gl: Record<string, unknown>;
  // definition end

  /** Constructor for **Human** library that is futher used for all operations
   * @param userConfig - user configuration object {@link Config}
   */
  constructor(userConfig?: Partial<Config>) {
    this.env = env;
    /*
    defaults.wasmPath = tf.version['tfjs-core'].includes('-') // custom build or official build
      ? 'https://vladmandic.github.io/tfjs/dist/'
      : `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${tf.version_core}/dist/`;
    */
    const tfVersion = (tf.version?.tfjs || tf.version_core).replace(/-(.*)/, '');
    defaults.wasmPath = `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${tfVersion}/dist/`;
    defaults.modelBasePath = env.browser ? '../models/' : 'file://models/';
    defaults.backend = env.browser ? 'humangl' : 'tensorflow';
    this.version = app.version; // expose version property on instance of class
    Object.defineProperty(this, 'version', { value: app.version }); // expose version property directly on class itself
    this.config = JSON.parse(JSON.stringify(defaults));
    Object.seal(this.config);
    this.config.cacheModels = typeof indexedDB !== 'undefined';
    if (userConfig) this.config = mergeDeep(this.config, userConfig);
    setModelLoadOptions(this.config);
    this.tf = tf;
    this.state = 'idle';
    this.#numTensors = 0;
    this.#analyzeMemoryLeaks = false;
    this.#checkSanity = false;
    this.performance = {};
    this.events = (typeof EventTarget !== 'undefined') ? new EventTarget() : undefined;
    // object that contains all initialized models
    this.models = new models.Models();
    // reexport draw methods
    this.result = { face: [], body: [], hand: [], gesture: [], object: [], performance: {}, timestamp: 0, persons: [], error: null };
    // export access to image processing
    // @ts-ignore eslint-typescript cannot correctly infer type in anonymous function
    this.process = { tensor: null, canvas: null };
    // export raw access to underlying models
    this.faceTriangulation = facemesh.triangulation;
    this.faceUVMap = facemesh.uvmap;
    // set gl info
    this.gl = humangl.config;
    // include platform info
    this.emit('create');
  }

  /** internal function to measure tensor leaks */
  analyze = (...msg: string[]) => {
    if (!this.#analyzeMemoryLeaks) return;
    const currentTensors = this.tf.engine().state.numTensors;
    const previousTensors = this.#numTensors;
    this.#numTensors = currentTensors;
    const leaked = currentTensors - previousTensors;
    if (leaked !== 0) log(...msg, leaked);
  };

  /** internal function for quick sanity check on inputs @hidden */
  #sanity = (input: Input): null | string => {
    if (!this.#checkSanity) return null;
    if (!input) return 'input is not defined';
    if (this.env.node && !(input instanceof tf.Tensor)) return 'input must be a tensor';
    try {
      this.tf.getBackend();
    } catch {
      return 'backend not loaded';
    }
    return null;
  };

  /** Reset configuration to default values */
  reset(): void {
    const currentBackend = this.config.backend; // save backend;
    this.config = JSON.parse(JSON.stringify(defaults));
    this.config.backend = currentBackend;
  }

  /** Validate current configuration schema */
  validate(userConfig?: Partial<Config>) {
    return validate(defaults, userConfig || this.config);
  }

  /** Exports face matching methods {@link match#similarity} */
  public similarity = match.similarity;
  /** Exports face matching methods {@link match#distance} */
  public distance = match.distance;
  /** Exports face matching methods {@link match#match} */
  public match = match.match;

  /** Utility wrapper for performance.now() */
  now(): number {
    return now();
  }

  /** Process input as return canvas and tensor
   *
   * @param input - any input {@link Input}
   * @param getTensor - should image processing also return tensor or just canvas
   * Returns object with `tensor` and `canvas`
   */
  image(input: Input, getTensor: boolean = true) {
    return image.process(input, this.config, getTensor);
  }

  /** Compare two input tensors for pixel simmilarity
   * - use `human.image` to process any valid input and get a tensor that can be used for compare
   * - when passing manually generated tensors:
   *  - both input tensors must be in format [1, height, width, 3]
   *  - if resolution of tensors does not match, second tensor will be resized to match resolution of the first tensor
   * - return value is pixel similarity score normalized by input resolution and rgb channels
  */
  compare(firstImageTensor: Tensor, secondImageTensor: Tensor): Promise<number> {
    return image.compare(this.config, firstImageTensor, secondImageTensor);
  }

  /** Explicit backend initialization
   *  - Normally done implicitly during initial load phase
   *  - Call to explictly register and initialize TFJS backend without any other operations
   *  - Use when changing backend during runtime
   */
  async init(): Promise<void> {
    await backend.check(this, true);
    await this.tf.ready();
  }

  /** Load method preloads all configured models on-demand
   * - Not explicitly required as any required model is load implicitly on it's first run
   *
   * @param userConfig - {@link Config}
  */
  async load(userConfig?: Partial<Config>): Promise<void> {
    this.state = 'load';
    const timeStamp = now();
    const count = Object.values(this.models).filter((model) => model).length;
    if (userConfig) this.config = mergeDeep(this.config, userConfig) as Config;

    if (this.env.initial) { // print version info on first run and check for correct backend setup
      if (this.config.debug) log(`version: ${this.version}`);
      if (this.config.debug) log(`tfjs version: ${this.tf.version['tfjs-core']}`);
      if (!await backend.check(this)) log('error: backend check failed');
      await tf.ready();
      if (this.env.browser) {
        if (this.config.debug) log('configuration:', this.config);
        if (this.config.debug) log('environment:', this.env);
        if (this.config.debug) log('tf flags:', this.tf.ENV['flags']);
      }
    }

    await models.load(this); // actually loads models
    if (this.env.initial && this.config.debug) log('tf engine state:', this.tf.engine().state.numBytes, 'bytes', this.tf.engine().state.numTensors, 'tensors'); // print memory stats on first run
    this.env.initial = false;

    const loaded = Object.values(this.models).filter((model) => model).length;
    if (loaded !== count) { // number of loaded models changed
      await models.validate(this); // validate kernel ops used by model against current backend
      this.emit('load');
    }

    const current = Math.trunc(now() - timeStamp);
    if (current > (this.performance.loadModels as number || 0)) this.performance.loadModels = this.env.perfadd ? (this.performance.loadModels || 0) + current : current;
  }

  /** emit event */
  emit = (event: string) => {
    if (this.events && this.events.dispatchEvent) this.events?.dispatchEvent(new Event(event));
  };

  /** Runs interpolation using last known result and returns smoothened result
   * Interpolation is based on time since last known result so can be called independently
   *
   * @param result - {@link Result} optional use specific result set to run interpolation on
   * @returns result - {@link Result}
   */
  next(result: Result = this.result): Result {
    return interpolate.calc(result, this.config) as Result;
  }

  /** get model loading/loaded stats */
  getModelStats(): ModelStats { return models.getModelStats(this); }

  /** Warmup method pre-initializes all configured models for faster inference
   * - can take significant time on startup
   * - only used for `webgl` and `humangl` backends
   * @param userConfig - {@link Config}
   * @returns result - {@link Result}
  */
  async warmup(userConfig?: Partial<Config>) {
    const t0 = now();
    const res = await warmups.warmup(this, userConfig);
    const t1 = now();
    this.performance.warmup = Math.trunc(t1 - t0);
    return res;
  }

  /** Run detect with tensorflow profiling
   * - result object will contain total exeuction time information for top-20 kernels
   * - actual detection object can be accessed via `human.result`
  */
  async profile(input: Input, userConfig?: Partial<Config>): Promise<Array<{ kernel: string, time: number, perc: number }>> {
    const profile = await this.tf.profile(() => this.detect(input, userConfig));
    const kernels: Record<string, number> = {};
    let total = 0;
    for (const kernel of profile.kernels) { // sum kernel time values per kernel
      if (kernels[kernel.name]) kernels[kernel.name] += kernel.kernelTimeMs;
      else kernels[kernel.name] = kernel.kernelTimeMs;
      total += kernel.kernelTimeMs;
    }
    const kernelArr: Array<{ kernel: string, time: number, perc: number }> = [];
    Object.entries(kernels).forEach((key) => kernelArr.push({ kernel: key[0], time: key[1] as unknown as number, perc: 0 })); // convert to array
    for (const kernel of kernelArr) {
      kernel.perc = Math.round(1000 * kernel.time / total) / 1000;
      kernel.time = Math.round(1000 * kernel.time) / 1000;
    }
    kernelArr.sort((a, b) => b.time - a.time); // sort
    kernelArr.length = 20; // crop
    return kernelArr;
  }

  /** Main detection method
   * - Analyze configuration: {@link Config}
   * - Pre-process input: {@link Input}
   * - Run inference for all configured models
   * - Process and return result: {@link Result}
   *
   * @param input - {@link Input}
   * @param userConfig - {@link Config}
   * @returns result - {@link Result}
  */
  async detect(input: Input, userConfig?: Partial<Config>): Promise<Result> {
    // detection happens inside a promise
    this.state = 'detect';
    return new Promise(async (resolve) => {
      this.state = 'config';
      let timeStamp;

      // update configuration
      this.config = mergeDeep(this.config, userConfig) as Config;

      // sanity checks
      this.state = 'check';
      const error = this.#sanity(input);
      if (error) {
        log(error, input);
        this.emit('error');
        resolve({ face: [], body: [], hand: [], gesture: [], object: [], performance: this.performance, timestamp: now(), persons: [], error });
      }

      const timeStart = now();

      // configure backend if needed
      await backend.check(this);

      // load models if enabled
      await this.load();

      timeStamp = now();
      this.state = 'image';
      const img = await image.process(input, this.config) as { canvas: AnyCanvas, tensor: Tensor };
      this.process = img;
      this.performance.inputProcess = this.env.perfadd ? (this.performance.inputProcess || 0) + Math.trunc(now() - timeStamp) : Math.trunc(now() - timeStamp);
      this.analyze('Get Image:');

      if (!img.tensor) {
        if (this.config.debug) log('could not convert input to tensor');
        this.emit('error');
        resolve({ face: [], body: [], hand: [], gesture: [], object: [], performance: this.performance, timestamp: now(), persons: [], error: 'could not convert input to tensor' });
        return;
      }
      this.emit('image');

      timeStamp = now();
      this.config.skipAllowed = await image.skip(this.config, img.tensor);
      if (!this.performance.totalFrames) this.performance.totalFrames = 0;
      if (!this.performance.cachedFrames) this.performance.cachedFrames = 0;
      (this.performance.totalFrames as number)++;
      if (this.config.skipAllowed) this.performance.cachedFrames++;
      this.performance.cacheCheck = this.env.perfadd ? (this.performance.cacheCheck || 0) + Math.trunc(now() - timeStamp) : Math.trunc(now() - timeStamp);
      this.analyze('Check Changed:');

      // prepare where to store model results
      // keep them with weak typing as it can be promise or not
      let faceRes: FaceResult[] | Promise<FaceResult[]> | never[] = [];
      let bodyRes: BodyResult[] | Promise<BodyResult[]> | never[] = [];
      let handRes: HandResult[] | Promise<HandResult[]> | never[] = [];
      let objectRes: ObjectResult[] | Promise<ObjectResult[]> | never[] = [];

      // run face detection followed by all models that rely on face bounding box: face mesh, age, gender, emotion
      this.state = 'detect:face';
      if (this.config.async) {
        faceRes = this.config.face.enabled ? face.detectFace(this, img.tensor) : [];
        if (this.performance.face) delete this.performance.face;
      } else {
        timeStamp = now();
        faceRes = this.config.face.enabled ? await face.detectFace(this, img.tensor) : [];
        this.performance.face = this.env.perfadd ? (this.performance.face || 0) + Math.trunc(now() - timeStamp) : Math.trunc(now() - timeStamp);
      }

      if (this.config.async && (this.config.body.maxDetected === -1 || this.config.hand.maxDetected === -1)) faceRes = await faceRes; // need face result for auto-detect number of hands or bodies

      // if async wait for results
      this.state = 'detect:await';
      if (this.config.async) [faceRes, bodyRes, handRes, objectRes] = await Promise.all([faceRes, bodyRes, handRes, objectRes]);

      this.performance.total = this.env.perfadd ? (this.performance.total || 0) + Math.trunc(now() - timeStart) : Math.trunc(now() - timeStart);
      const shape = this.process?.tensor?.shape || [];
      this.result = {
        face: faceRes as FaceResult[],
        body: bodyRes as BodyResult[],
        hand: handRes as HandResult[],
        gesture: [],
        object: objectRes as ObjectResult[],
        performance: this.performance,
        canvas: this.process.canvas,
        timestamp: Date.now(),
        error: null,
        get persons() { return persons.join(faceRes as FaceResult[], bodyRes as BodyResult[], handRes as HandResult[], [], shape); },
      };

      // finally dispose input tensor
      tf.dispose(img.tensor);

      // log('Result:', result);
      this.emit('detect');
      this.state = 'idle';
      resolve(this.result);
    });
  }
}

/** Class Human as default export */
/* eslint no-restricted-exports: ["off", { "restrictedNamedExports": ["default"] }] */
export { Human as default, match, models };
